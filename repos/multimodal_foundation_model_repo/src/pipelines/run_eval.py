"""Evaluation pipeline for THREADS reproduction.

This module executes the design-locked eval stage:
- load manifest records and exported embeddings
- build/load split manifests
- evaluate classification/grading/survival tasks
- optionally evaluate retrieval/prompting tasks
- run statistical analyses and persist artifacts

Public entrypoint:
- ``run_eval(cfg_or_path: Any = "configs/default.yaml") -> dict[str, Any]``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.core.config import ExperimentConfig
from src.core.logging_utils import (
    UnifiedLogger,
    capture_exception,
    finalize_run,
    init_unified_logger,
)
from src.data.manifest_schema import ManifestRecord
from src.data.manifest_store import ManifestStore
from src.data.split_manager import SplitManager, SplitValidationError
from src.eval.linear_probe import LinearProbeEvaluator
from src.eval.prompting_eval import PromptingEvaluator
from src.eval.retrieval_eval import RetrievalEvaluator
from src.eval.stats_tests import StatsAnalyzer
from src.eval.survival_eval import SurvivalEvaluator, resolve_alpha
from src.utils.io import read_parquet, write_json, write_parquet
from src.utils.seeding import seed_everything


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "configs/default.yaml"
DEFAULT_STAGE: str = "eval"

DEFAULT_SEED: int = 42
DEFAULT_DETERMINISTIC: bool = True

DEFAULT_MANIFEST_PATH: str = "data/processed/manifests/pretrain_public_merged.parquet"
DEFAULT_SPLITS_ROOT: str = "data/processed/splits"
DEFAULT_EMBEDDINGS_ROOT: str = "data/processed/embeddings"
DEFAULT_SLIDE_EMBEDDINGS_FILE: str = "threads_slide_embeddings.parquet"
DEFAULT_PATIENT_EMBEDDINGS_FILE: str = "threads_patient_embeddings.parquet"
DEFAULT_MOLECULAR_EMBEDDINGS_FILE: str = "threads_molecular_embeddings.parquet"

DEFAULT_MODEL_NAME: str = "THREADS"
DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_GROUP_BY: str = "patient_id"
DEFAULT_STRATIFY_BY_LABEL: str = "label"

DEFAULT_C_VALUE: float = 0.5
DEFAULT_SOLVER: str = "lbfgs"
DEFAULT_MAX_ITER_CLASSIFICATION: int = 10000
DEFAULT_CLASS_WEIGHT: str = "balanced"
DEFAULT_MAX_ITER_SURVIVAL: int = 10000

DEFAULT_CV_FOLDS: int = 5
DEFAULT_MC_SPLITS: int = 50
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_BOOTSTRAP_REPLICATES: int = 100

DEFAULT_RETRIEVAL_METRIC: str = "l2"
DEFAULT_RETRIEVAL_TOP_K: Tuple[int, ...] = (1, 5, 10)

DEFAULT_FAIL_ON_TASK_ERROR: bool = False
DEFAULT_FAIL_ON_MISSING_EMBEDDINGS: bool = True
DEFAULT_ENABLE_FEWSHOT: bool = True
DEFAULT_FEWSHOT_K_VALUES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)

DEFAULT_OUTPUT_FOLD_FILE: str = "eval_fold_metrics.parquet"
DEFAULT_OUTPUT_TASK_SUMMARY_FILE: str = "eval_task_summary.parquet"
DEFAULT_OUTPUT_ANOVA_FILE: str = "eval_stats_anova.parquet"
DEFAULT_OUTPUT_TUKEY_FILE: str = "eval_stats_tukey.parquet"
DEFAULT_OUTPUT_MIXED_FILE: str = "eval_stats_mixed_effects.parquet"
DEFAULT_OUTPUT_SKIPPED_FILE: str = "eval_skipped_tasks.parquet"
DEFAULT_OUTPUT_SPLITS_DIR: str = "eval_effective_splits"
DEFAULT_OUTPUT_SUMMARY_FILE: str = "eval_summary.json"

TASK_TYPE_BINARY: str = "binary_classification"
TASK_TYPE_SUBTYPING: str = "subtyping_multiclass"
TASK_TYPE_GRADING: str = "grading_multiclass"
TASK_TYPE_SURVIVAL: str = "survival"

UNIT_SLIDE: str = "slide"
UNIT_PATIENT: str = "patient"

METRIC_MACRO_AUC: str = "macro_auc"
METRIC_BACC: str = "balanced_accuracy"
METRIC_QWK: str = "quadratic_weighted_kappa"
METRIC_CINDEX: str = "c_index"
METRIC_MAP_AT_PREFIX: str = "map_at_"


class EvalPipelineError(Exception):
    """Base exception for eval stage failures."""


class EvalConfigError(EvalPipelineError):
    """Raised when runtime eval configuration is invalid."""


class EvalRuntimeError(EvalPipelineError):
    """Raised when eval execution fails."""


@dataclass(frozen=True)
class _RuntimeOptions:
    """Resolved eval runtime options."""

    manifest_path: str
    splits_root: str
    slide_embeddings_path: str
    patient_embeddings_path: str
    molecular_embeddings_path: str
    fail_on_task_error: bool
    fail_on_missing_embeddings: bool
    enable_fewshot: bool
    fewshot_k_values: Tuple[int, ...]
    model_name: str


@dataclass(frozen=True)
class _TaskSpec:
    """Flattened task specification."""

    family: str
    task_name: str
    cohort: str
    unit: str
    task_type: str
    metric: str
    split_strategy: str
    survival_endpoint: str
    enabled: bool


class EvalPipeline:
    """Stage runner for THREADS downstream evaluation."""

    def __init__(self, config: ExperimentConfig, logger: UnifiedLogger) -> None:
        self._config: ExperimentConfig = self._validate_config(config)
        self._cfg_dict: Dict[str, Any] = self._config.to_dict()
        self._logger: UnifiedLogger = logger

        self._opts: _RuntimeOptions = self._resolve_runtime_options(self._cfg_dict)
        self._validate_paper_invariants(self._cfg_dict)

        self._manifest_store: ManifestStore = ManifestStore(self._opts.manifest_path)
        self._split_manager: SplitManager = SplitManager(
            split_dir=self._opts.splits_root,
            seed=_resolve_seed_from_cfg(self._cfg_dict),
        )
        self._stats: StatsAnalyzer = StatsAnalyzer(
            alpha=0.05,
            bootstrap_replicates=DEFAULT_BOOTSTRAP_REPLICATES,
            random_seed=_resolve_seed_from_cfg(self._cfg_dict),
        )

    def run(self) -> Dict[str, Any]:
        """Execute eval stage and return summary payload."""
        stage_start: float = perf_counter()

        all_records: List[ManifestRecord] = self._manifest_store.load()
        all_records = sorted(all_records, key=lambda item: (item.patient_id, item.sample_id))
        if len(all_records) == 0:
            raise EvalRuntimeError(f"Manifest has zero records: {self._opts.manifest_path}")

        slide_embeddings: pd.DataFrame = self._load_embeddings(
            self._opts.slide_embeddings_path,
            unit=UNIT_SLIDE,
        )
        patient_embeddings: pd.DataFrame = self._load_embeddings(
            self._opts.patient_embeddings_path,
            unit=UNIT_PATIENT,
        )

        task_specs: List[_TaskSpec] = self._load_task_specs(self._cfg_dict)
        if len(task_specs) == 0:
            raise EvalConfigError("No enabled downstream tasks were found in config.")

        self._logger.log_event(
            "eval_stage_started",
            payload={
                "manifest_path": self._opts.manifest_path,
                "record_count": len(all_records),
                "task_count": len(task_specs),
                "slide_embeddings_path": self._opts.slide_embeddings_path,
                "patient_embeddings_path": self._opts.patient_embeddings_path,
                "splits_root": self._opts.splits_root,
            },
        )

        fold_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        sample_to_patient: Dict[str, str] = {
            record.sample_id: record.patient_id for record in all_records
        }

        for task_spec in task_specs:
            task_start: float = perf_counter()
            try:
                task_records: List[ManifestRecord] = self._collect_task_records(
                    all_records=all_records,
                    task_spec=task_spec,
                )
                if len(task_records) == 0:
                    skipped_rows.append(
                        self._build_skipped_row(
                            task_spec=task_spec,
                            reason="no_task_records",
                            details="No records matched task_name/cohort filters.",
                        )
                    )
                    continue

                splits: List[Dict[str, Any]] = self._resolve_task_splits(
                    task_spec=task_spec,
                    task_records=task_records,
                )
                self._persist_effective_splits(task_spec=task_spec, splits=splits)

                embedding_table: pd.DataFrame = (
                    patient_embeddings if task_spec.unit == UNIT_PATIENT else slide_embeddings
                )
                task_fold_rows: List[Dict[str, Any]] = self._evaluate_task_with_splits(
                    task_spec=task_spec,
                    task_records=task_records,
                    splits=splits,
                    embedding_table=embedding_table,
                    sample_to_patient=sample_to_patient,
                    method_type="linear_probe" if task_spec.task_type != TASK_TYPE_SURVIVAL else "survival_probe",
                    k_shot=None,
                )
                fold_rows.extend(task_fold_rows)

                if (
                    self._opts.enable_fewshot
                    and task_spec.task_type in {TASK_TYPE_BINARY, TASK_TYPE_SUBTYPING, TASK_TYPE_GRADING}
                ):
                    for split_obj in splits:
                        for k_value in self._opts.fewshot_k_values:
                            try:
                                fewshot_split: Dict[str, Any] = self._split_manager.make_fewshot(
                                    base_split=split_obj,
                                    k_per_class=int(k_value),
                                )
                            except SplitValidationError:
                                # Config policy: omit infeasible k.
                                continue
                            fewshot_rows: List[Dict[str, Any]] = self._evaluate_task_with_splits(
                                task_spec=task_spec,
                                task_records=task_records,
                                splits=[fewshot_split],
                                embedding_table=embedding_table,
                                sample_to_patient=sample_to_patient,
                                method_type="linear_probe_fewshot",
                                k_shot=int(k_value),
                            )
                            fold_rows.extend(fewshot_rows)

                self._logger.log_event(
                    "eval_task_completed",
                    payload={
                        "task_name": task_spec.task_name,
                        "family": task_spec.family,
                        "unit": task_spec.unit,
                        "task_type": task_spec.task_type,
                        "elapsed_sec": perf_counter() - task_start,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                skipped_rows.append(
                    self._build_skipped_row(
                        task_spec=task_spec,
                        reason=type(exc).__name__,
                        details=str(exc),
                    )
                )
                self._logger.log_event(
                    "eval_task_failed",
                    payload={
                        "task_name": task_spec.task_name,
                        "family": task_spec.family,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                if self._opts.fail_on_task_error:
                    raise

        retrieval_rows: List[Dict[str, Any]] = self._evaluate_retrieval_tasks(
            all_records=all_records,
            slide_embeddings=slide_embeddings,
            sample_to_patient=sample_to_patient,
        )
        fold_rows.extend(retrieval_rows)

        prompting_rows, prompting_skips = self._evaluate_prompting_tasks(
            all_records=all_records,
            slide_embeddings=slide_embeddings,
            patient_embeddings=patient_embeddings,
            sample_to_patient=sample_to_patient,
        )
        fold_rows.extend(prompting_rows)
        skipped_rows.extend(prompting_skips)

        fold_df: pd.DataFrame = pd.DataFrame(fold_rows)
        if fold_df.empty:
            raise EvalRuntimeError("No evaluation fold metrics were produced.")
        fold_df = self._normalize_metrics_dataframe(fold_df)

        summary_df: pd.DataFrame = self._aggregate_task_summary(fold_df)
        stats_inputs_df: pd.DataFrame = fold_df[fold_df["k_shot"].isna()].copy()

        anova_df: pd.DataFrame
        tukey_df: pd.DataFrame
        mixed_df: pd.DataFrame
        if stats_inputs_df.empty:
            anova_df = pd.DataFrame()
            tukey_df = pd.DataFrame()
            mixed_df = pd.DataFrame()
        else:
            anova_df = self._stats.anova_two_way(stats_inputs_df)
            tukey_df = self._stats.tukey_hsd(stats_inputs_df)
            mixed_df = self._stats.mixed_effects(stats_inputs_df)

        fold_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_FOLD_FILE
        summary_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_TASK_SUMMARY_FILE
        anova_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_ANOVA_FILE
        tukey_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_TUKEY_FILE
        mixed_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_MIXED_FILE
        skipped_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_SKIPPED_FILE

        write_parquet(fold_df, fold_path, sort_columns=True)
        write_parquet(summary_df, summary_path, sort_columns=True)
        write_parquet(anova_df, anova_path, sort_columns=True)
        write_parquet(tukey_df, tukey_path, sort_columns=True)
        write_parquet(mixed_df, mixed_path, sort_columns=True)
        write_parquet(pd.DataFrame(skipped_rows), skipped_path, sort_columns=True)

        summary: Dict[str, Any] = {
            "stage": DEFAULT_STAGE,
            "run_id": self._logger.run_id,
            "elapsed_sec": perf_counter() - stage_start,
            "model_name": self._opts.model_name,
            "task_count_configured": len(task_specs),
            "fold_metrics_count": int(fold_df.shape[0]),
            "summary_rows_count": int(summary_df.shape[0]),
            "skipped_count": len(skipped_rows),
            "fold_metrics_path": str(fold_path),
            "task_summary_path": str(summary_path),
            "anova_path": str(anova_path),
            "tukey_path": str(tukey_path),
            "mixed_effects_path": str(mixed_path),
            "skipped_tasks_path": str(skipped_path),
        }

        summary_output_path: Path = self._logger.paths.metrics / DEFAULT_OUTPUT_SUMMARY_FILE
        write_json(summary, summary_output_path)
        self._logger.log_event("eval_stage_completed", payload=summary)
        return summary

    def _collect_task_records(
        self,
        all_records: Sequence[ManifestRecord],
        task_spec: _TaskSpec,
    ) -> List[ManifestRecord]:
        output: List[ManifestRecord] = []
        for record in all_records:
            if task_spec.task_name not in record.task_labels:
                continue
            if task_spec.cohort != "" and str(record.cohort) != task_spec.cohort:
                continue
            output.append(record)
        return output

    def _resolve_task_splits(
        self,
        task_spec: _TaskSpec,
        task_records: Sequence[ManifestRecord],
    ) -> List[Dict[str, Any]]:
        if len(task_records) == 0:
            return []

        strategy: str = task_spec.split_strategy.strip().lower()
        if strategy == "official_single_fold":
            split_obj: Dict[str, Any] = self._split_manager.load_official(task_spec.task_name)
            return [split_obj]
        if strategy == "cv5_80_20":
            return self._split_manager.make_cv(
                records=list(task_records),
                n_folds=DEFAULT_CV_FOLDS,
                stratify_by=task_spec.task_name,
                group_by=DEFAULT_GROUP_BY,
            )
        if strategy == "mc50":
            return self._split_manager.make_monte_carlo(
                records=list(task_records),
                n_splits=DEFAULT_MC_SPLITS,
                test_size=DEFAULT_TEST_SIZE,
                stratify_by=task_spec.task_name,
                group_by=DEFAULT_GROUP_BY,
            )
        raise EvalConfigError(f"Unsupported split strategy: {task_spec.split_strategy!r}")

    def _persist_effective_splits(self, task_spec: _TaskSpec, splits: Sequence[Mapping[str, Any]]) -> None:
        if len(splits) == 0:
            return
        output_dir: Path = self._logger.paths.splits / DEFAULT_OUTPUT_SPLITS_DIR / task_spec.task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, split_obj in enumerate(splits):
            split_name: str = f"split_{idx:03d}.json"
            split_path: Path = output_dir / split_name
            write_json(
                {
                    "task_name": task_spec.task_name,
                    "family": task_spec.family,
                    "split": dict(split_obj),
                },
                split_path,
            )

    def _evaluate_task_with_splits(
        self,
        task_spec: _TaskSpec,
        task_records: Sequence[ManifestRecord],
        splits: Sequence[Mapping[str, Any]],
        embedding_table: pd.DataFrame,
        sample_to_patient: Mapping[str, str],
        method_type: str,
        k_shot: Optional[int],
    ) -> List[Dict[str, Any]]:
        id_to_embedding: Dict[str, np.ndarray] = self._build_embedding_index(embedding_table, unit=task_spec.unit)
        label_maps: Dict[str, Any] = self._build_label_maps(
            task_spec=task_spec,
            task_records=task_records,
        )

        output_rows: List[Dict[str, Any]] = []
        for split_obj in splits:
            fold_id: str = str(split_obj.get("fold_id", "0"))
            train_ids_raw: List[str] = [str(item) for item in split_obj.get("train_ids", [])]
            test_ids_raw: List[str] = [str(item) for item in split_obj.get("test_ids", [])]
            if len(train_ids_raw) == 0 or len(test_ids_raw) == 0:
                continue

            train_ids: List[str] = self._resolve_eval_ids(
                raw_ids=train_ids_raw,
                unit=task_spec.unit,
                sample_to_patient=sample_to_patient,
            )
            test_ids: List[str] = self._resolve_eval_ids(
                raw_ids=test_ids_raw,
                unit=task_spec.unit,
                sample_to_patient=sample_to_patient,
            )
            if len(train_ids) == 0 or len(test_ids) == 0:
                continue

            x_train: np.ndarray = self._stack_embeddings(train_ids, id_to_embedding=id_to_embedding)
            x_test: np.ndarray = self._stack_embeddings(test_ids, id_to_embedding=id_to_embedding)

            if task_spec.task_type == TASK_TYPE_SURVIVAL:
                time_map: Mapping[str, float] = label_maps["time_map"]
                event_map: Mapping[str, bool] = label_maps["event_map"]
                y_train_time: np.ndarray = self._collect_numeric_targets(train_ids, target_map=time_map)
                y_train_event: np.ndarray = self._collect_bool_targets(train_ids, target_map=event_map)
                y_test_time: np.ndarray = self._collect_numeric_targets(test_ids, target_map=time_map)
                y_test_event: np.ndarray = self._collect_bool_targets(test_ids, target_map=event_map)

                alpha_value: float = resolve_alpha(
                    task_name=self._build_survival_alpha_task_name(task_spec),
                    model_name=self._opts.model_name,
                )
                evaluator_survival: SurvivalEvaluator = SurvivalEvaluator(
                    alpha=float(alpha_value),
                    max_iter=DEFAULT_MAX_ITER_SURVIVAL,
                )
                evaluator_survival.fit(
                    x_train=x_train,
                    y_time=y_train_time,
                    y_event=y_train_event,
                )
                risk: np.ndarray = np.asarray(evaluator_survival.predict_risk(x_test), dtype=np.float64)
                metric_value: float = evaluator_survival.score_c_index(
                    y_time=y_test_time,
                    y_event=y_test_event,
                    risk=risk,
                )
                output_rows.append(
                    self._build_metric_row(
                        task_spec=task_spec,
                        fold_id=fold_id,
                        metric_name=METRIC_CINDEX,
                        value=metric_value,
                        method_type=method_type,
                        k_shot=k_shot,
                        alpha_used=float(alpha_value),
                    )
                )
                continue

            label_map: Mapping[str, Any] = label_maps["label_map"]
            y_train: np.ndarray = self._collect_targets(train_ids, target_map=label_map)
            y_test: np.ndarray = self._collect_targets(test_ids, target_map=label_map)

            evaluator_lp: LinearProbeEvaluator = LinearProbeEvaluator(
                c_value=DEFAULT_C_VALUE,
                max_iter=DEFAULT_MAX_ITER_CLASSIFICATION,
                solver=DEFAULT_SOLVER,
                class_weight=DEFAULT_CLASS_WEIGHT,
            )
            evaluator_lp.fit(x_train=x_train, y_train=y_train)
            probability: np.ndarray = np.asarray(evaluator_lp.predict_proba(x_test), dtype=np.float64)

            if task_spec.task_type == TASK_TYPE_BINARY:
                metric_value_binary: float = evaluator_lp.score_binary_auc(
                    y_true=y_test,
                    y_prob=probability,
                )
                output_rows.append(
                    self._build_metric_row(
                        task_spec=task_spec,
                        fold_id=fold_id,
                        metric_name=METRIC_MACRO_AUC,
                        value=metric_value_binary,
                        method_type=method_type,
                        k_shot=k_shot,
                        alpha_used=None,
                    )
                )
                continue

            class_labels: np.ndarray = np.asarray(np.unique(y_train), dtype=object)
            predicted_indices: np.ndarray = np.argmax(probability, axis=1).astype(np.int64, copy=False)
            y_pred: np.ndarray = class_labels[predicted_indices]

            if task_spec.task_type == TASK_TYPE_SUBTYPING:
                metric_value_bacc: float = evaluator_lp.score_multiclass_bacc(
                    y_true=y_test,
                    y_pred=y_pred,
                )
                output_rows.append(
                    self._build_metric_row(
                        task_spec=task_spec,
                        fold_id=fold_id,
                        metric_name=METRIC_BACC,
                        value=metric_value_bacc,
                        method_type=method_type,
                        k_shot=k_shot,
                        alpha_used=None,
                    )
                )
                continue

            if task_spec.task_type == TASK_TYPE_GRADING:
                metric_value_qwk: float = evaluator_lp.score_qwk(
                    y_true=y_test,
                    y_pred=y_pred,
                )
                output_rows.append(
                    self._build_metric_row(
                        task_spec=task_spec,
                        fold_id=fold_id,
                        metric_name=METRIC_QWK,
                        value=metric_value_qwk,
                        method_type=method_type,
                        k_shot=k_shot,
                        alpha_used=None,
                    )
                )
                continue

            raise EvalConfigError(f"Unsupported task_type: {task_spec.task_type!r}")

        return output_rows

    def _evaluate_retrieval_tasks(
        self,
        all_records: Sequence[ManifestRecord],
        slide_embeddings: pd.DataFrame,
        sample_to_patient: Mapping[str, str],
    ) -> List[Dict[str, Any]]:
        retrieval_cfg: Mapping[str, Any] = _deep_get(self._cfg_dict, ("downstream_public", "retrieval"), {})
        if not _as_bool(retrieval_cfg.get("enabled", False), default=False):
            return []

        evaluations: Sequence[Mapping[str, Any]] = retrieval_cfg.get("evaluations", [])
        if not isinstance(evaluations, Sequence):
            return []

        id_to_embedding: Dict[str, np.ndarray] = self._build_embedding_index(slide_embeddings, unit=UNIT_SLIDE)
        top_k_values: Tuple[int, ...] = _as_int_tuple(
            retrieval_cfg.get("top_k", list(DEFAULT_RETRIEVAL_TOP_K)),
            default=DEFAULT_RETRIEVAL_TOP_K,
        )
        evaluator: RetrievalEvaluator = RetrievalEvaluator(
            metric=DEFAULT_RETRIEVAL_METRIC,
            top_k=list(top_k_values),
        )

        output_rows: List[Dict[str, Any]] = []
        for item in evaluations:
            if not isinstance(item, Mapping):
                continue
            if not _as_bool(item.get("enabled", False), default=False):
                continue
            task_name: str = _as_str(item.get("task_name", ""), default="")
            if task_name == "":
                continue
            dataset_key: str = _as_str(item.get("dataset", ""), default="")
            label_type: str = _as_str(item.get("label_type", ""), default="")

            label_map: Dict[str, str] = self._build_retrieval_label_map(
                all_records=all_records,
                dataset_key=dataset_key,
                label_type=label_type,
            )
            if len(label_map) <= max(top_k_values):
                continue

            record_subset: List[ManifestRecord] = [
                record for record in all_records if record.sample_id in label_map
            ]
            if len(record_subset) < 4:
                continue

            stratify_key: str = self._resolve_retrieval_stratify_key(label_type=label_type)
            try:
                splits: List[Dict[str, Any]] = self._split_manager.make_cv(
                    records=record_subset,
                    n_folds=DEFAULT_CV_FOLDS,
                    stratify_by=stratify_key,
                    group_by=DEFAULT_GROUP_BY,
                )
            except Exception:
                continue

            for split_obj in splits:
                fold_id: str = str(split_obj.get("fold_id", "0"))
                train_ids: List[str] = self._resolve_eval_ids(
                    raw_ids=[str(x) for x in split_obj.get("train_ids", [])],
                    unit=UNIT_SLIDE,
                    sample_to_patient=sample_to_patient,
                )
                test_ids: List[str] = self._resolve_eval_ids(
                    raw_ids=[str(x) for x in split_obj.get("test_ids", [])],
                    unit=UNIT_SLIDE,
                    sample_to_patient=sample_to_patient,
                )
                train_ids = [item_id for item_id in train_ids if item_id in label_map and item_id in id_to_embedding]
                test_ids = [item_id for item_id in test_ids if item_id in label_map and item_id in id_to_embedding]
                if len(train_ids) <= max(top_k_values) or len(test_ids) == 0:
                    continue

                x_ref: np.ndarray = self._stack_embeddings(train_ids, id_to_embedding)
                y_ref: np.ndarray = np.asarray([label_map[item_id] for item_id in train_ids], dtype=object)
                x_query: np.ndarray = self._stack_embeddings(test_ids, id_to_embedding)
                y_query: np.ndarray = np.asarray([label_map[item_id] for item_id in test_ids], dtype=object)

                evaluator.build_index(x_ref=x_ref, y_ref=y_ref)
                query_output: Mapping[str, Any] = evaluator.query(x_q=x_query)
                for k_value in top_k_values:
                    map_value: float = float(
                        evaluator.map_at_k(
                            y_q=y_query,
                            retrieved_labels=query_output,
                            k=int(k_value),
                        )
                    )
                    output_rows.append(
                        {
                            "task": task_name,
                            "dataset": dataset_key if dataset_key != "" else "retrieval",
                            "family": "retrieval",
                            "fold": fold_id,
                            "metric_name": f"{METRIC_MAP_AT_PREFIX}{int(k_value)}",
                            "value": map_value,
                            "model_name": self._opts.model_name,
                            "method_type": "retrieval",
                            "embedding_level": UNIT_SLIDE,
                            "k_shot": np.nan,
                            "alpha_used": np.nan,
                            "unit": UNIT_SLIDE,
                            "task_type": "retrieval",
                        }
                    )
        return output_rows

    def _evaluate_prompting_tasks(
        self,
        all_records: Sequence[ManifestRecord],
        slide_embeddings: pd.DataFrame,
        patient_embeddings: pd.DataFrame,
        sample_to_patient: Mapping[str, str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        prompting_cfg: Mapping[str, Any] = _deep_get(self._cfg_dict, ("downstream_public", "molecular_prompting"), {})
        if not _as_bool(prompting_cfg.get("enabled", False), default=False):
            return [], []

        molecular_path: Path = Path(self._opts.molecular_embeddings_path).expanduser().resolve()
        if not molecular_path.exists():
            skip_row: Dict[str, Any] = {
                "task": "molecular_prompting",
                "family": "prompting",
                "reason": "molecular_embeddings_missing",
                "details": str(molecular_path),
            }
            return [], [skip_row]

        molecular_df: pd.DataFrame = read_parquet(molecular_path)
        if "embedding" not in molecular_df.columns:
            skip_row = {
                "task": "molecular_prompting",
                "family": "prompting",
                "reason": "invalid_molecular_schema",
                "details": "Column 'embedding' missing.",
            }
            return [], [skip_row]

        # Generic prompting evaluation is optional and data-dependent.
        # We run a minimal nearest-prompt classification evaluation for tasks where
        # labels and both molecular/WSI embeddings are available with matching sample_id.
        evaluator: PromptingEvaluator = PromptingEvaluator()
        slide_index: Dict[str, np.ndarray] = self._build_embedding_index(slide_embeddings, unit=UNIT_SLIDE)
        patient_index: Dict[str, np.ndarray] = self._build_embedding_index(patient_embeddings, unit=UNIT_PATIENT)
        molecular_index: Dict[str, np.ndarray] = self._build_generic_embedding_index(
            frame=molecular_df,
            id_col_candidates=("sample_id", "patient_id"),
        )

        output_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        task_specs: List[_TaskSpec] = self._load_task_specs(self._cfg_dict)
        for task_spec in task_specs:
            if task_spec.task_type not in {TASK_TYPE_BINARY, TASK_TYPE_SUBTYPING, TASK_TYPE_GRADING}:
                continue
            task_records: List[ManifestRecord] = self._collect_task_records(all_records, task_spec)
            if len(task_records) < 4:
                continue

            support_ids_raw: List[str] = [item.sample_id for item in task_records if item.sample_id in molecular_index]
            if task_spec.unit == UNIT_PATIENT:
                support_ids: List[str] = self._resolve_eval_ids(
                    raw_ids=support_ids_raw,
                    unit=UNIT_PATIENT,
                    sample_to_patient=sample_to_patient,
                )
            else:
                support_ids = sorted(set(support_ids_raw))

            label_map: Dict[str, Any]
            if task_spec.unit == UNIT_PATIENT:
                label_map = self._build_patient_label_map(task_records, task_spec.task_name)
            else:
                label_map = {item.sample_id: item.task_labels.get(task_spec.task_name, "") for item in task_records}

            support_ids = [item_id for item_id in support_ids if item_id in label_map and str(label_map[item_id]).strip() != ""]
            if len(support_ids) < 4:
                continue

            if task_spec.unit == UNIT_PATIENT:
                z_wsi_map: Dict[str, np.ndarray] = patient_index
                z_mol_ids: List[str] = []
                for patient_id in support_ids:
                    patient_samples: List[str] = [s for s, p in sample_to_patient.items() if p == patient_id and s in molecular_index]
                    if len(patient_samples) == 0:
                        continue
                    z_mol_ids.append(patient_samples[0])
                if len(z_mol_ids) < 4:
                    continue
                z_mol: np.ndarray = self._stack_embeddings(z_mol_ids, molecular_index)
                y_support: np.ndarray = np.asarray(
                    [label_map[sample_to_patient[item_id]] if item_id in sample_to_patient else label_map[item_id] for item_id in z_mol_ids],
                    dtype=object,
                )
                query_ids: List[str] = [item_id for item_id in support_ids if item_id in z_wsi_map]
                z_query: np.ndarray = self._stack_embeddings(query_ids, z_wsi_map)
                y_query: np.ndarray = np.asarray([label_map[item_id] for item_id in query_ids], dtype=object)
            else:
                z_wsi_map = slide_index
                query_ids = [item_id for item_id in support_ids if item_id in z_wsi_map]
                z_query = self._stack_embeddings(query_ids, z_wsi_map)
                y_query = np.asarray([label_map[item_id] for item_id in query_ids], dtype=object)
                z_mol = self._stack_embeddings(query_ids, molecular_index)
                y_support = y_query

            if len(query_ids) == 0:
                continue
            prompts: Dict[str, object] = evaluator.build_molecular_prompts(z_mol=z_mol, y=y_support)
            prediction: Mapping[str, Any] = evaluator.classify_by_nearest_prompt(
                z_wsi=z_query,
                prompts=prompts,
            )
            y_pred: np.ndarray = np.asarray(prediction["pred_labels"], dtype=object)

            # Prompting classification metric follows task routing.
            if task_spec.task_type == TASK_TYPE_BINARY:
                # Use proxy probabilities via inverse distances for AUC.
                distances: np.ndarray = np.asarray(prediction["distances"], dtype=np.float64)
                class_order: List[str] = [str(x) for x in prediction["class_order"]]
                if len(class_order) != 2:
                    skipped_rows.append(
                        self._build_skipped_row(
                            task_spec=task_spec,
                            reason="prompt_binary_requires_two_classes",
                            details=f"class_order={class_order}",
                        )
                    )
                    continue
                # Convert distances to score: higher score => positive class.
                positive_index: int = 1
                positive_score: np.ndarray = -distances[:, positive_index]
                lp_eval: LinearProbeEvaluator = LinearProbeEvaluator()
                metric_value: float = lp_eval.score_binary_auc(y_true=y_query, y_prob=positive_score)
                metric_name: str = METRIC_MACRO_AUC
            elif task_spec.task_type == TASK_TYPE_SUBTYPING:
                lp_eval = LinearProbeEvaluator()
                metric_value = lp_eval.score_multiclass_bacc(y_true=y_query, y_pred=y_pred)
                metric_name = METRIC_BACC
            else:
                lp_eval = LinearProbeEvaluator()
                metric_value = lp_eval.score_qwk(y_true=y_query, y_pred=y_pred)
                metric_name = METRIC_QWK

            output_rows.append(
                {
                    "task": task_spec.task_name,
                    "dataset": task_spec.cohort if task_spec.cohort != "" else "prompting",
                    "family": task_spec.family,
                    "fold": "prompting",
                    "metric_name": metric_name,
                    "value": float(metric_value),
                    "model_name": self._opts.model_name,
                    "method_type": "molecular_prompting",
                    "embedding_level": task_spec.unit,
                    "k_shot": np.nan,
                    "alpha_used": np.nan,
                    "unit": task_spec.unit,
                    "task_type": task_spec.task_type,
                }
            )

        return output_rows, skipped_rows

    def _build_retrieval_label_map(
        self,
        all_records: Sequence[ManifestRecord],
        dataset_key: str,
        label_type: str,
    ) -> Dict[str, str]:
        output: Dict[str, str] = {}
        for record in all_records:
            if not self._record_matches_retrieval_dataset(record=record, dataset_key=dataset_key):
                continue

            if label_type == "cancer_type":
                label_value: str = str(record.cohort)
            elif label_type == "fine_subtype":
                label_value = str(record.task_labels.get("EBRAINS_fine_subtyping", "")).strip()
            elif label_type == "coarse_subtype":
                label_value = str(record.task_labels.get("EBRAINS_coarse_subtyping", "")).strip()
            else:
                label_value = ""

            if label_value == "":
                continue
            output[record.sample_id] = label_value
        return output

    def _resolve_retrieval_stratify_key(self, label_type: str) -> str:
        if label_type == "fine_subtype":
            return "EBRAINS_fine_subtyping"
        if label_type == "coarse_subtype":
            return "EBRAINS_coarse_subtyping"
        if label_type == "cancer_type":
            return "cohort"
        return "cohort"

    def _record_matches_retrieval_dataset(self, record: ManifestRecord, dataset_key: str) -> bool:
        normalized_key: str = dataset_key.strip().upper()
        cohort_name: str = str(record.cohort).strip().upper()
        if normalized_key == "":
            return True
        if normalized_key == "EBRAINS":
            return cohort_name == "EBRAINS"
        if normalized_key == "CPTAC_10_COHORTS":
            return cohort_name.startswith("CPTAC")
        return cohort_name == normalized_key

    def _build_label_maps(
        self,
        task_spec: _TaskSpec,
        task_records: Sequence[ManifestRecord],
    ) -> Dict[str, Any]:
        if task_spec.task_type != TASK_TYPE_SURVIVAL:
            if task_spec.unit == UNIT_PATIENT:
                label_map: Dict[str, Any] = self._build_patient_label_map(
                    task_records=task_records,
                    task_name=task_spec.task_name,
                )
            else:
                label_map = {
                    item.sample_id: item.task_labels.get(task_spec.task_name, "")
                    for item in task_records
                }
            return {"label_map": label_map}

        if task_spec.unit == UNIT_PATIENT:
            time_map, event_map = self._build_patient_survival_maps(task_records, task_spec)
        else:
            time_map = {}
            event_map = {}
            for item in task_records:
                time_value, event_value = self._resolve_survival_targets(item, task_spec)
                time_map[item.sample_id] = time_value
                event_map[item.sample_id] = event_value
        return {"time_map": time_map, "event_map": event_map}

    def _build_patient_label_map(
        self,
        task_records: Sequence[ManifestRecord],
        task_name: str,
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {}
        for item in task_records:
            patient_id: str = item.patient_id
            label_value: Any = item.task_labels.get(task_name, "")
            if patient_id not in output:
                output[patient_id] = label_value
            elif output[patient_id] != label_value:
                raise EvalRuntimeError(
                    "Inconsistent patient-level labels for task "
                    f"{task_name!r}, patient_id={patient_id!r}: "
                    f"{output[patient_id]!r} vs {label_value!r}"
                )
        return output

    def _build_patient_survival_maps(
        self,
        task_records: Sequence[ManifestRecord],
        task_spec: _TaskSpec,
    ) -> Tuple[Dict[str, float], Dict[str, bool]]:
        time_map: Dict[str, float] = {}
        event_map: Dict[str, bool] = {}
        for item in task_records:
            patient_id: str = item.patient_id
            time_value, event_value = self._resolve_survival_targets(item, task_spec)
            if patient_id not in time_map:
                time_map[patient_id] = time_value
                event_map[patient_id] = event_value
            else:
                if not np.isclose(float(time_map[patient_id]), float(time_value)):
                    raise EvalRuntimeError(
                        "Inconsistent survival time for patient-level task "
                        f"{task_spec.task_name!r}, patient_id={patient_id!r}."
                    )
                if bool(event_map[patient_id]) != bool(event_value):
                    raise EvalRuntimeError(
                        "Inconsistent survival event for patient-level task "
                        f"{task_spec.task_name!r}, patient_id={patient_id!r}."
                    )
        return time_map, event_map

    def _resolve_survival_targets(
        self,
        record: ManifestRecord,
        task_spec: _TaskSpec,
    ) -> Tuple[float, bool]:
        endpoint: str = task_spec.survival_endpoint.strip().lower()
        endpoint_aliases: List[str] = [endpoint] if endpoint != "" else []
        if endpoint == "overall_survival":
            endpoint_aliases.extend(["os", "overall"])
        if endpoint == "progression_free_survival":
            endpoint_aliases.extend(["pfs", "progression_free", "progression-free"])

        candidate_time_keys: List[str] = []
        candidate_event_keys: List[str] = []
        for alias in endpoint_aliases:
            candidate_time_keys.extend(
                [
                    f"{task_spec.task_name}_{alias}_time",
                    f"{task_spec.task_name}__{alias}_time",
                    f"{alias}_time",
                    f"{alias}_duration",
                ]
            )
            candidate_event_keys.extend(
                [
                    f"{task_spec.task_name}_{alias}_event",
                    f"{task_spec.task_name}__{alias}_event",
                    f"{alias}_event",
                    f"{alias}_status",
                ]
            )

        candidate_time_keys.extend(
            [
                f"{task_spec.task_name}_time",
                f"{task_spec.task_name}__time",
                "time",
                "survival_time",
                "duration",
            ]
        )
        candidate_event_keys.extend(
            [
                f"{task_spec.task_name}_event",
                f"{task_spec.task_name}__event",
                "event",
                "status",
                "censor",
                "censored",
            ]
        )

        time_value: Optional[float] = _first_numeric_value(
            mappings=(record.task_labels, record.meta),
            keys=candidate_time_keys,
        )
        event_value_raw: Optional[Any] = _first_value(
            mappings=(record.task_labels, record.meta),
            keys=candidate_event_keys,
        )

        if time_value is None:
            raise EvalRuntimeError(
                f"Missing survival time for task={task_spec.task_name}, sample_id={record.sample_id}."
            )
        if event_value_raw is None:
            raise EvalRuntimeError(
                f"Missing survival event for task={task_spec.task_name}, sample_id={record.sample_id}."
            )
        event_value: bool = _to_bool_event(event_value_raw)
        return float(time_value), bool(event_value)

    def _build_metric_row(
        self,
        task_spec: _TaskSpec,
        fold_id: str,
        metric_name: str,
        value: float,
        method_type: str,
        k_shot: Optional[int],
        alpha_used: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "task": task_spec.task_name,
            "dataset": task_spec.cohort if task_spec.cohort != "" else "unknown",
            "family": task_spec.family,
            "fold": str(fold_id),
            "metric_name": str(metric_name),
            "value": float(value),
            "model_name": self._opts.model_name,
            "method_type": str(method_type),
            "embedding_level": task_spec.unit,
            "k_shot": float(k_shot) if k_shot is not None else np.nan,
            "alpha_used": float(alpha_used) if alpha_used is not None else np.nan,
            "unit": task_spec.unit,
            "task_type": task_spec.task_type,
        }

    def _build_skipped_row(
        self,
        task_spec: _TaskSpec,
        reason: str,
        details: str,
    ) -> Dict[str, Any]:
        return {
            "task": task_spec.task_name,
            "dataset": task_spec.cohort,
            "family": task_spec.family,
            "reason": str(reason),
            "details": str(details),
        }

    def _normalize_metrics_dataframe(self, frame: pd.DataFrame) -> pd.DataFrame:
        required_columns: Tuple[str, ...] = (
            "task",
            "dataset",
            "family",
            "fold",
            "metric_name",
            "value",
            "model_name",
            "method_type",
            "embedding_level",
            "k_shot",
            "alpha_used",
            "unit",
            "task_type",
        )
        missing: List[str] = [column for column in required_columns if column not in frame.columns]
        if len(missing) > 0:
            raise EvalRuntimeError(f"Fold metrics table missing required columns: {missing}")

        frame = frame.copy()
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame = frame.dropna(subset=["value"]).copy()
        if frame.empty:
            raise EvalRuntimeError("Fold metrics table is empty after numeric coercion.")
        if not np.isfinite(np.asarray(frame["value"], dtype=np.float64)).all():
            raise EvalRuntimeError("Fold metrics contain non-finite values.")

        for column in ("task", "dataset", "family", "fold", "metric_name", "model_name", "method_type", "embedding_level", "unit", "task_type"):
            frame[column] = frame[column].astype(str).str.strip()

        frame["k_shot"] = pd.to_numeric(frame["k_shot"], errors="coerce")
        frame["alpha_used"] = pd.to_numeric(frame["alpha_used"], errors="coerce")
        frame = frame.sort_values(
            by=["task", "method_type", "metric_name", "fold"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        return frame

    def _aggregate_task_summary(self, fold_df: pd.DataFrame) -> pd.DataFrame:
        group_cols: List[str] = [
            "task",
            "dataset",
            "family",
            "metric_name",
            "model_name",
            "method_type",
            "embedding_level",
            "unit",
            "task_type",
            "k_shot",
        ]
        rows: List[Dict[str, Any]] = []

        grouped = fold_df.groupby(group_cols, dropna=False, sort=True)
        for keys, group in grouped:
            values: np.ndarray = np.asarray(group["value"], dtype=np.float64)
            n_values: int = int(values.shape[0])
            mean_value: float = float(np.mean(values))
            if n_values > 1:
                se_value: float = float(np.std(values, ddof=1) / np.sqrt(float(n_values)))
                ci_low: float = float(mean_value - se_value)
                ci_high: float = float(mean_value + se_value)
            else:
                se_value = float("nan")
                ci_low, ci_high = self._stats.bootstrap_ci(values=values, n_boot=DEFAULT_BOOTSTRAP_REPLICATES)

            row: Dict[str, Any] = dict(zip(group_cols, keys))
            row.update(
                {
                    "mean": mean_value,
                    "se": se_value,
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "n_folds": n_values,
                }
            )
            rows.append(row)

        summary_df: pd.DataFrame = pd.DataFrame(rows)
        if summary_df.empty:
            return summary_df
        summary_df = summary_df.sort_values(
            by=["task", "method_type", "metric_name", "k_shot"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)
        return summary_df

    def _load_embeddings(self, path: str, unit: str) -> pd.DataFrame:
        resolved: Path = Path(path).expanduser().resolve()
        if not resolved.exists():
            if self._opts.fail_on_missing_embeddings:
                raise EvalConfigError(f"Embedding file does not exist: {resolved}")
            return pd.DataFrame()

        frame: pd.DataFrame = read_parquet(resolved)
        if frame.empty:
            if self._opts.fail_on_missing_embeddings:
                raise EvalRuntimeError(f"Embedding file is empty: {resolved}")
            return frame

        if "embedding" not in frame.columns:
            raise EvalRuntimeError(f"Embedding file missing 'embedding' column: {resolved}")
        if "embedding_dim" in frame.columns:
            dims: np.ndarray = pd.to_numeric(frame["embedding_dim"], errors="coerce").to_numpy()
            if np.any(dims != DEFAULT_EMBEDDING_DIM):
                raise EvalRuntimeError(
                    f"Embedding dim mismatch in {resolved}; expected {DEFAULT_EMBEDDING_DIM}."
                )

        id_column: str = "patient_id" if unit == UNIT_PATIENT else "sample_id"
        if id_column not in frame.columns:
            raise EvalRuntimeError(f"Embedding file missing id column '{id_column}': {resolved}")

        frame = frame.copy()
        frame[id_column] = frame[id_column].astype(str).str.strip()
        frame = frame[frame[id_column] != ""].copy()
        frame = frame.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
        return frame

    def _build_embedding_index(self, embedding_table: pd.DataFrame, unit: str) -> Dict[str, np.ndarray]:
        id_column: str = "patient_id" if unit == UNIT_PATIENT else "sample_id"
        if embedding_table.empty:
            return {}
        if id_column not in embedding_table.columns:
            raise EvalRuntimeError(f"Embedding table missing required id column: {id_column}")
        if "embedding" not in embedding_table.columns:
            raise EvalRuntimeError("Embedding table missing 'embedding' column.")

        output: Dict[str, np.ndarray] = {}
        for _, row in embedding_table.iterrows():
            item_id: str = str(row[id_column]).strip()
            if item_id == "":
                continue
            vector: np.ndarray = np.asarray(row["embedding"], dtype=np.float64).reshape(-1)
            if vector.shape[0] != DEFAULT_EMBEDDING_DIM:
                raise EvalRuntimeError(
                    f"Embedding width mismatch for id={item_id}; expected {DEFAULT_EMBEDDING_DIM}, got {vector.shape[0]}."
                )
            if not np.isfinite(vector).all():
                raise EvalRuntimeError(f"Non-finite embedding encountered for id={item_id}.")
            output[item_id] = vector
        return output

    def _build_generic_embedding_index(
        self,
        frame: pd.DataFrame,
        id_col_candidates: Sequence[str],
    ) -> Dict[str, np.ndarray]:
        if frame.empty or "embedding" not in frame.columns:
            return {}
        selected_col: Optional[str] = None
        for col in id_col_candidates:
            if col in frame.columns:
                selected_col = col
                break
        if selected_col is None:
            return {}

        output: Dict[str, np.ndarray] = {}
        for _, row in frame.iterrows():
            item_id: str = str(row[selected_col]).strip()
            if item_id == "":
                continue
            vector: np.ndarray = np.asarray(row["embedding"], dtype=np.float64).reshape(-1)
            if vector.shape[0] != DEFAULT_EMBEDDING_DIM:
                continue
            if not np.isfinite(vector).all():
                continue
            output[item_id] = vector
        return output

    def _stack_embeddings(self, ids: Sequence[str], id_to_embedding: Mapping[str, np.ndarray]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        missing_ids: List[str] = []
        for item_id in ids:
            if item_id not in id_to_embedding:
                missing_ids.append(item_id)
                continue
            vectors.append(np.asarray(id_to_embedding[item_id], dtype=np.float64))

        if len(missing_ids) > 0 and self._opts.fail_on_missing_embeddings:
            raise EvalRuntimeError(
                "Missing embeddings for required IDs. "
                f"count={len(missing_ids)} preview={missing_ids[:20]}"
            )
        if len(vectors) == 0:
            raise EvalRuntimeError("No embeddings available after ID filtering.")
        matrix: np.ndarray = np.vstack(vectors).astype(np.float64, copy=False)
        return matrix

    def _collect_targets(self, ids: Sequence[str], target_map: Mapping[str, Any]) -> np.ndarray:
        values: List[Any] = []
        missing_ids: List[str] = []
        for item_id in ids:
            if item_id not in target_map:
                missing_ids.append(item_id)
                continue
            value: Any = target_map[item_id]
            if str(value).strip() == "":
                missing_ids.append(item_id)
                continue
            values.append(value)
        if len(missing_ids) > 0:
            raise EvalRuntimeError(
                "Missing labels for required IDs. "
                f"count={len(missing_ids)} preview={missing_ids[:20]}"
            )
        return np.asarray(values, dtype=object)

    def _collect_numeric_targets(self, ids: Sequence[str], target_map: Mapping[str, float]) -> np.ndarray:
        values: List[float] = []
        missing_ids: List[str] = []
        for item_id in ids:
            if item_id not in target_map:
                missing_ids.append(item_id)
                continue
            values.append(float(target_map[item_id]))
        if len(missing_ids) > 0:
            raise EvalRuntimeError(
                "Missing numeric targets for required IDs. "
                f"count={len(missing_ids)} preview={missing_ids[:20]}"
            )
        output: np.ndarray = np.asarray(values, dtype=np.float64)
        if not np.isfinite(output).all():
            raise EvalRuntimeError("Numeric targets contain NaN/Inf.")
        return output

    def _collect_bool_targets(self, ids: Sequence[str], target_map: Mapping[str, bool]) -> np.ndarray:
        values: List[bool] = []
        missing_ids: List[str] = []
        for item_id in ids:
            if item_id not in target_map:
                missing_ids.append(item_id)
                continue
            values.append(bool(target_map[item_id]))
        if len(missing_ids) > 0:
            raise EvalRuntimeError(
                "Missing boolean targets for required IDs. "
                f"count={len(missing_ids)} preview={missing_ids[:20]}"
            )
        return np.asarray(values, dtype=bool)

    def _resolve_eval_ids(
        self,
        raw_ids: Sequence[str],
        unit: str,
        sample_to_patient: Mapping[str, str],
    ) -> List[str]:
        if unit == UNIT_SLIDE:
            unique_ids: List[str] = sorted({str(item).strip() for item in raw_ids if str(item).strip() != ""})
            return unique_ids
        if unit == UNIT_PATIENT:
            patient_ids: List[str] = []
            for sample_id in raw_ids:
                normalized_sample_id: str = str(sample_id).strip()
                if normalized_sample_id == "":
                    continue
                if normalized_sample_id not in sample_to_patient:
                    continue
                patient_ids.append(str(sample_to_patient[normalized_sample_id]))
            return sorted(set(patient_ids))
        raise EvalConfigError(f"Unsupported unit: {unit!r}")

    def _build_survival_alpha_task_name(self, task_spec: _TaskSpec) -> str:
        # Align task naming convention expected by alpha override resolver.
        task_name_normalized: str = task_spec.task_name.replace("_", " ").strip()
        endpoint_normalized: str = task_spec.survival_endpoint.replace("_", " ").strip()
        composed: str = f"{task_name_normalized} {endpoint_normalized}".strip()
        return composed

    def _load_task_specs(self, cfg: Mapping[str, Any]) -> List[_TaskSpec]:
        family_map: Mapping[str, Any] = _deep_get(cfg, ("downstream_public", "task_families"), {})
        if not isinstance(family_map, Mapping):
            raise EvalConfigError("downstream_public.task_families must be a mapping.")

        specs: List[_TaskSpec] = []
        for family_name, family_payload in family_map.items():
            if not isinstance(family_payload, Mapping):
                continue
            family_enabled: bool = _as_bool(family_payload.get("enabled", True), default=True)
            if not family_enabled:
                continue
            tasks_payload: Any = family_payload.get("public_tasks", [])
            if not isinstance(tasks_payload, Sequence):
                continue
            for task_item in tasks_payload:
                if not isinstance(task_item, Mapping):
                    continue
                task_enabled: bool = _as_bool(task_item.get("enabled", True), default=True)
                if not task_enabled:
                    continue
                task_name: str = _as_str(task_item.get("task_name", ""), default="")
                if task_name == "":
                    continue
                task_spec: _TaskSpec = _TaskSpec(
                    family=str(family_name),
                    task_name=task_name,
                    cohort=_as_str(task_item.get("cohort", ""), default=""),
                    unit=_normalize_unit(_as_str(task_item.get("unit", UNIT_SLIDE), default=UNIT_SLIDE)),
                    task_type=_normalize_task_type(
                        _as_str(task_item.get("task_type", TASK_TYPE_BINARY), default=TASK_TYPE_BINARY)
                    ),
                    metric=_as_str(task_item.get("metric", ""), default=""),
                    split_strategy=_as_str(task_item.get("split_strategy", "cv5_80_20"), default="cv5_80_20"),
                    survival_endpoint=_as_str(task_item.get("survival_endpoint", ""), default=""),
                    enabled=True,
                )
                specs.append(task_spec)

        specs = sorted(specs, key=lambda item: (item.family, item.task_name))
        return specs

    def _resolve_runtime_options(self, cfg: Mapping[str, Any]) -> _RuntimeOptions:
        manifest_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "manifest_path"),
                    ("runtime", "manifest_path"),
                    ("pretrain_public", "pretrain_public", "manifests", "files", "merged_public"),
                ),
                default=DEFAULT_MANIFEST_PATH,
            ),
            default=DEFAULT_MANIFEST_PATH,
        )

        splits_root: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "splits_root"),
                    ("runtime", "splits_root"),
                    ("downstream_public", "io", "splits_root"),
                ),
                default=DEFAULT_SPLITS_ROOT,
            ),
            default=DEFAULT_SPLITS_ROOT,
        )

        embeddings_root: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "embeddings_root"),
                    ("runtime", "embeddings_root"),
                    ("downstream_public", "io", "embeddings_root"),
                    ("pretrain_public", "pretrain_public", "io_roots", "embeddings_root"),
                ),
                default=DEFAULT_EMBEDDINGS_ROOT,
            ),
            default=DEFAULT_EMBEDDINGS_ROOT,
        )
        embeddings_root_path: Path = Path(embeddings_root).expanduser().resolve()

        slide_embeddings_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "slide_embeddings_path"),
                    ("runtime", "slide_embeddings_path"),
                ),
                default=str(embeddings_root_path / DEFAULT_SLIDE_EMBEDDINGS_FILE),
            ),
            default=str(embeddings_root_path / DEFAULT_SLIDE_EMBEDDINGS_FILE),
        )
        patient_embeddings_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "patient_embeddings_path"),
                    ("runtime", "patient_embeddings_path"),
                ),
                default=str(embeddings_root_path / DEFAULT_PATIENT_EMBEDDINGS_FILE),
            ),
            default=str(embeddings_root_path / DEFAULT_PATIENT_EMBEDDINGS_FILE),
        )
        molecular_embeddings_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "molecular_embeddings_path"),
                    ("runtime", "molecular_embeddings_path"),
                ),
                default=str(embeddings_root_path / DEFAULT_MOLECULAR_EMBEDDINGS_FILE),
            ),
            default=str(embeddings_root_path / DEFAULT_MOLECULAR_EMBEDDINGS_FILE),
        )

        fewshot_enabled: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "enable_fewshot"),
                    ("downstream_public", "split_policy", "few_shot", "enabled"),
                ),
                default=DEFAULT_ENABLE_FEWSHOT,
            ),
            default=DEFAULT_ENABLE_FEWSHOT,
        )
        fewshot_values: Tuple[int, ...] = _as_int_tuple(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "fewshot_k_values"),
                    ("downstream_public", "split_policy", "few_shot", "k_values"),
                    ("evaluation", "few_shot", "k_values"),
                ),
                default=list(DEFAULT_FEWSHOT_K_VALUES),
            ),
            default=DEFAULT_FEWSHOT_K_VALUES,
        )

        fail_on_task_error: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "fail_on_task_error"),
                    ("runtime", "fail_on_task_error"),
                ),
                default=DEFAULT_FAIL_ON_TASK_ERROR,
            ),
            default=DEFAULT_FAIL_ON_TASK_ERROR,
        )
        fail_on_missing_embeddings: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "fail_on_missing_embeddings"),
                    ("runtime", "fail_on_missing_embeddings"),
                ),
                default=DEFAULT_FAIL_ON_MISSING_EMBEDDINGS,
            ),
            default=DEFAULT_FAIL_ON_MISSING_EMBEDDINGS,
        )

        model_name: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "eval", "model_name"),
                    ("model", "name"),
                ),
                default=DEFAULT_MODEL_NAME,
            ),
            default=DEFAULT_MODEL_NAME,
        )

        return _RuntimeOptions(
            manifest_path=manifest_path,
            splits_root=splits_root,
            slide_embeddings_path=slide_embeddings_path,
            patient_embeddings_path=patient_embeddings_path,
            molecular_embeddings_path=molecular_embeddings_path,
            fail_on_task_error=fail_on_task_error,
            fail_on_missing_embeddings=fail_on_missing_embeddings,
            enable_fewshot=fewshot_enabled,
            fewshot_k_values=fewshot_values,
            model_name=model_name,
        )

    def _validate_paper_invariants(self, cfg: Mapping[str, Any]) -> None:
        c_value: float = _as_float(
            _first_present(
                cfg,
                (
                    ("linear_probe", "classification", "C"),
                    ("downstream_public", "linear_probe", "classification", "C"),
                ),
                default=DEFAULT_C_VALUE,
            ),
            default=DEFAULT_C_VALUE,
        )
        solver: str = _as_str(
            _first_present(
                cfg,
                (
                    ("linear_probe", "classification", "solver"),
                    ("downstream_public", "linear_probe", "classification", "solver"),
                ),
                default=DEFAULT_SOLVER,
            ),
            default=DEFAULT_SOLVER,
        ).lower()
        max_iter_cls: int = _as_int(
            _first_present(
                cfg,
                (
                    ("linear_probe", "classification", "max_iter"),
                    ("downstream_public", "linear_probe", "classification", "max_iter"),
                ),
                default=DEFAULT_MAX_ITER_CLASSIFICATION,
            ),
            default=DEFAULT_MAX_ITER_CLASSIFICATION,
        )
        class_weight: str = _as_str(
            _first_present(
                cfg,
                (
                    ("linear_probe", "classification", "class_weight"),
                    ("downstream_public", "linear_probe", "classification", "class_weight"),
                ),
                default=DEFAULT_CLASS_WEIGHT,
            ),
            default=DEFAULT_CLASS_WEIGHT,
        ).lower()
        max_iter_survival: int = _as_int(
            _first_present(
                cfg,
                (
                    ("linear_probe", "survival", "max_iter"),
                    ("downstream_public", "linear_probe", "survival", "max_iter"),
                ),
                default=DEFAULT_MAX_ITER_SURVIVAL,
            ),
            default=DEFAULT_MAX_ITER_SURVIVAL,
        )
        embedding_dim: int = _as_int(
            _first_present(
                cfg,
                (
                    ("model", "slide_embedding_dim"),
                    ("downstream_public", "shared_constants", "embedding_dim"),
                ),
                default=DEFAULT_EMBEDDING_DIM,
            ),
            default=DEFAULT_EMBEDDING_DIM,
        )

        if not np.isclose(c_value, DEFAULT_C_VALUE):
            raise EvalConfigError(f"linear_probe.classification.C must be {DEFAULT_C_VALUE}, got {c_value}.")
        if solver != DEFAULT_SOLVER:
            raise EvalConfigError(f"linear_probe.classification.solver must be '{DEFAULT_SOLVER}', got {solver!r}.")
        if max_iter_cls != DEFAULT_MAX_ITER_CLASSIFICATION:
            raise EvalConfigError(
                f"linear_probe.classification.max_iter must be {DEFAULT_MAX_ITER_CLASSIFICATION}, got {max_iter_cls}."
            )
        if class_weight != DEFAULT_CLASS_WEIGHT:
            raise EvalConfigError(
                f"linear_probe.classification.class_weight must be '{DEFAULT_CLASS_WEIGHT}', got {class_weight!r}."
            )
        if max_iter_survival != DEFAULT_MAX_ITER_SURVIVAL:
            raise EvalConfigError(
                f"linear_probe.survival.max_iter must be {DEFAULT_MAX_ITER_SURVIVAL}, got {max_iter_survival}."
            )
        if embedding_dim != DEFAULT_EMBEDDING_DIM:
            raise EvalConfigError(
                f"model.slide_embedding_dim must be {DEFAULT_EMBEDDING_DIM}, got {embedding_dim}."
            )

    def _validate_config(self, config: ExperimentConfig) -> ExperimentConfig:
        if not isinstance(config, ExperimentConfig):
            raise EvalConfigError(f"config must be ExperimentConfig, got {type(config).__name__}.")
        if str(config.stage).strip().lower() != DEFAULT_STAGE:
            raise EvalConfigError(f"Expected stage='{DEFAULT_STAGE}', got '{config.stage}'.")
        return config


def run_eval(cfg_or_path: Any = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Run eval stage end-to-end."""
    config: ExperimentConfig = _resolve_experiment_config(cfg_or_path)
    logger: UnifiedLogger = init_unified_logger(cfg=config, stage=DEFAULT_STAGE)
    seed: int = _resolve_seed_from_cfg(config.to_dict())
    deterministic: bool = _resolve_deterministic_from_cfg(config.to_dict())

    try:
        seed_state: Dict[str, Any] = seed_everything(
            global_seed=seed,
            deterministic=deterministic,
            stage=DEFAULT_STAGE,
        )
        logger.log_event("seed_initialized", payload=seed_state)

        pipeline: EvalPipeline = EvalPipeline(config=config, logger=logger)
        summary: Dict[str, Any] = pipeline.run()

        finalize_run(logger=logger, status="success", summary=summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        capture_exception(logger=logger, exc=exc, stage_step="run_eval")
        finalize_run(
            logger=logger,
            status="failed",
            summary={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------
def _resolve_experiment_config(cfg_or_path: Any) -> ExperimentConfig:
    if isinstance(cfg_or_path, ExperimentConfig):
        cfg_or_path.validate()
        return cfg_or_path
    if isinstance(cfg_or_path, str):
        config_path: str = str(cfg_or_path).strip() or DEFAULT_CONFIG_PATH
        config: ExperimentConfig = ExperimentConfig.from_yaml(config_path)
        config.validate()
        return config
    raise EvalConfigError(f"Unsupported cfg_or_path type: {type(cfg_or_path).__name__}.")


def _deep_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _first_present(cfg: Mapping[str, Any], paths: Sequence[Tuple[str, ...]], default: Any) -> Any:
    for path in paths:
        value: Any = _deep_get(cfg, path, None)
        if value is not None:
            return value
    return default


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return str(default)
    normalized: str = str(value).strip()
    if normalized == "":
        return str(default)
    return normalized


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise EvalConfigError("Integer value cannot be bool.")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not float(value).is_integer():
            raise EvalConfigError(f"Expected integer-like value, got {value!r}.")
        return int(value)
    if isinstance(value, str):
        token: str = value.strip()
        if token == "":
            return int(default)
        return int(token)
    raise EvalConfigError(f"Expected int value, got {type(value).__name__}.")


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        raise EvalConfigError("Float value cannot be bool.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token: str = value.strip()
        if token == "":
            return float(default)
        return float(token)
    raise EvalConfigError(f"Expected float value, got {type(value).__name__}.")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        token: str = value.strip().lower()
        if token in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if token in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise EvalConfigError(f"Expected bool value, got {value!r}.")


def _as_int_tuple(value: Any, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output: List[int] = []
        for item in value:
            output.append(_as_int(item, default=1))
        if len(output) == 0:
            return default
        return tuple(sorted(set(output)))
    raise EvalConfigError(f"Expected sequence[int], got {type(value).__name__}.")


def _resolve_seed_from_cfg(cfg: Mapping[str, Any]) -> int:
    seed_value: Any = _first_present(
        cfg,
        (
            ("runtime", "seed"),
            ("train_pretrain", "pretrain", "training", "seed"),
            ("pretraining", "training", "seed"),
            ("downstream_public", "split_policy", "seed"),
        ),
        default=DEFAULT_SEED,
    )
    return _as_int(seed_value, default=DEFAULT_SEED)


def _resolve_deterministic_from_cfg(cfg: Mapping[str, Any]) -> bool:
    deterministic_value: Any = _first_present(
        cfg,
        (
            ("runtime", "deterministic"),
            ("train_pretrain", "pretrain", "training", "deterministic"),
        ),
        default=DEFAULT_DETERMINISTIC,
    )
    return _as_bool(deterministic_value, default=DEFAULT_DETERMINISTIC)


def _normalize_unit(unit_value: str) -> str:
    normalized: str = unit_value.strip().lower()
    if normalized in {"slide", "wsi"}:
        return UNIT_SLIDE
    if normalized in {"patient", "case"}:
        return UNIT_PATIENT
    raise EvalConfigError(f"Unsupported task unit: {unit_value!r}.")


def _normalize_task_type(task_type_value: str) -> str:
    normalized: str = task_type_value.strip().lower()
    if normalized in {TASK_TYPE_BINARY, TASK_TYPE_SUBTYPING, TASK_TYPE_GRADING, TASK_TYPE_SURVIVAL}:
        return normalized
    raise EvalConfigError(f"Unsupported task_type: {task_type_value!r}.")


def _first_value(mappings: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Optional[Any]:
    for key in keys:
        for mapping in mappings:
            if key in mapping:
                value: Any = mapping[key]
                if value is None:
                    continue
                if str(value).strip() == "":
                    continue
                return value
    return None


def _first_numeric_value(mappings: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        for mapping in mappings:
            if key not in mapping:
                continue
            value: Any = mapping[key]
            try:
                numeric: float = float(value)
            except Exception:
                continue
            if np.isfinite(numeric):
                return float(numeric)
    return None


def _to_bool_event(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not np.isfinite(float(value)):
            raise EvalRuntimeError(f"Invalid survival event numeric value: {value!r}")
        return bool(int(float(value)))
    token: str = str(value).strip().lower()
    if token in {"1", "true", "t", "yes", "y", "event", "dead", "deceased"}:
        return True
    if token in {"0", "false", "f", "no", "n", "censored", "alive"}:
        return False
    raise EvalRuntimeError(f"Unsupported survival event value: {value!r}")


__all__ = [
    "EvalPipelineError",
    "EvalConfigError",
    "EvalRuntimeError",
    "EvalPipeline",
    "run_eval",
]
