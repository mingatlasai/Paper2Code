"""Evaluation orchestration script for TITAN/TITAN V checkpoints.

This runner is intentionally thin. It coordinates:
- config loading and validation,
- deterministic runtime setup,
- checkpoint capability gating,
- embedding extraction/reuse,
- task dispatch to design-locked evaluators,
- consolidated reporting artifacts.

It does not re-implement evaluation algorithms; those live in `src/eval/*`.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.core.config_schema import ConfigLoadError, ConfigValidationError, ExperimentConfig, ModelConfig
from src.core.registry import Registry
from src.core.utils import (
    ensure_dir,
    load_checkpoint,
    seed_everything,
    validate_repro_constants,
    write_csv,
    write_json,
)
from src.data.datasets import BaseDataset
from src.eval.embed_api import EmbeddingService
from src.eval.few_shot import FewShotEvaluator
from src.eval.knn_probe import KNNProbeEvaluator
from src.eval.linear_probe import LinearProbeEvaluator
from src.eval.report_generation import ReportGenerationEvaluator
from src.eval.retrieval import RetrievalEvaluator
from src.eval.statistics import GLMMFitResult, StatsAnalyzer
from src.eval.survival import SurvivalEvaluator
from src.eval.zero_shot import ZeroShotEvaluator
from src.models.coca_multimodal import CoCaModel
from src.models.titan_encoder import TITANEncoder


# -----------------------------------------------------------------------------
# Config-locked constants from provided config/task contract.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DEFAULT_CONFIG_PATH: str = "config.yaml"
_FALLBACK_CONFIG_PATH: str = "configs/eval/linear_probe.yaml"

_DEFAULT_EVAL_REPORT_NAME: str = "eval_report.json"

_DEFAULT_TASKS_ALL_ORDER: Tuple[str, ...] = (
    "linear_probe",
    "knn_probe",
    "few_shot",
    "zero_shot",
    "retrieval",
    "cross_modal_retrieval",
    "report_generation",
    "survival",
)

_DEFAULT_SPLIT_COL: str = "split"


class RunEvalError(RuntimeError):
    """Base exception for evaluation orchestration failures."""


@dataclass(frozen=True)
class EvalCliArgs:
    """CLI arguments for evaluation execution."""

    config: str
    ckpt: Optional[str]
    tasks: str
    metadata_csv: Optional[str]
    output_dir: Optional[str]
    seed: Optional[int]
    bootstrap_n: Optional[int]
    few_shot_runs: Optional[int]
    strict_task_fail: bool
    save_embeddings: bool


@dataclass(frozen=True)
class CheckpointInfo:
    """Resolved checkpoint metadata."""

    path: str
    has_multimodal_keys: bool
    state_key_count: int


@dataclass(frozen=True)
class SlideTable:
    """Canonical slide-level evaluation table."""

    frame: pd.DataFrame
    slide_id_col: str
    label_col: Optional[str]
    split_col: str
    fold_col: Optional[str]
    time_col: Optional[str]
    event_col: Optional[str]
    report_col: Optional[str]


class EvalOrchestrator:
    """Config-driven evaluation orchestrator."""

    def __init__(self, args: EvalCliArgs) -> None:
        self.args: EvalCliArgs = args
        self.cfg: ExperimentConfig = self._load_and_override_config(args)

        validate_repro_constants(self.cfg)

        seed_info: Dict[str, Any] = seed_everything(
            seed=int(self.cfg.runtime.seed),
            deterministic=bool(self.cfg.runtime.deterministic),
        )
        self.seed_info: Dict[str, Any] = seed_info

        self.output_dir: Path = ensure_dir(
            self.args.output_dir if self.args.output_dir else self.cfg.paths.output_root
        )
        self.eval_output_dir: Path = ensure_dir(self.output_dir / "eval")

        self.registry: Registry = Registry(auto_register_defaults=True)

        self.stats: StatsAnalyzer = StatsAnalyzer(
            bootstrap_default=int(self.cfg.evaluation.bootstrap_samples)
        )

        self.tasks_requested: List[str] = self._parse_tasks(self.args.tasks)

        self.ckpt_path: Path = self._resolve_checkpoint_path(
            explicit_ckpt=self.args.ckpt,
            cfg=self.cfg,
        )
        self.ckpt_info: CheckpointInfo = self._inspect_checkpoint(self.ckpt_path)

        self._validate_global_invariants()

        needs_language: bool = self._requires_language_capabilities(self.tasks_requested)
        if needs_language and not self.ckpt_info.has_multimodal_keys:
            requested: str = ", ".join(self.tasks_requested)
            raise RunEvalError(
                "Requested language-capable task(s) with a vision-only checkpoint. "
                f"checkpoint={self.ckpt_info.path}, tasks={requested}."
            )

        self.vision_model, self.multimodal_model = self._build_models_and_load_checkpoint(
            needs_multimodal=needs_language
        )
        self.embedding_service: EmbeddingService = self._build_embedding_service(
            vision_model=self.vision_model,
            multimodal_model=self.multimodal_model,
        )

        self.slide_table: SlideTable = self._load_slide_table(self.args.metadata_csv)
        self.slide_embeddings: Optional[np.ndarray] = None

    def run(self) -> Dict[str, Any]:
        """Run all requested tasks and return consolidated report payload."""
        tasks_completed: List[str] = []
        tasks_failed: List[Dict[str, str]] = []
        results: Dict[str, Any] = {}

        for task_name in self.tasks_requested:
            try:
                task_result: Dict[str, Any] = self._run_single_task(task_name)
                results[task_name] = task_result
                tasks_completed.append(task_name)
                self._write_task_artifact(task_name=task_name, payload=task_result)
            except Exception as exc:  # noqa: BLE001
                failure_record: Dict[str, str] = {
                    "task": task_name,
                    "error": str(exc),
                }
                tasks_failed.append(failure_record)
                if self.args.strict_task_fail:
                    raise

        stats_payload: Dict[str, Any] = self._build_statistics_block(results)

        report_payload: Dict[str, Any] = {
            "run_metadata": {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "seed": int(self.cfg.runtime.seed),
                "config_path": str(self.cfg.paths.project_root),
                "checkpoint_path": str(self.ckpt_path),
                "checkpoint_has_multimodal": bool(self.ckpt_info.has_multimodal_keys),
                "checkpoint_state_key_count": int(self.ckpt_info.state_key_count),
            },
            "config_snapshot": {
                "mode": self.cfg.mode,
                "stage": self.cfg.stage,
                "patch_size": int(self.cfg.data.patch_size),
                "magnification": str(self.cfg.data.magnification),
                "feature_dim": int(self.cfg.data.feature_dim),
                "evaluation": self.cfg.evaluation.model_dump(),
            },
            "seed_info": self.seed_info,
            "tasks_requested": list(self.tasks_requested),
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "results": results,
            "statistics": stats_payload,
        }

        report_path: Path = self.eval_output_dir / _DEFAULT_EVAL_REPORT_NAME
        write_json(report_payload, report_path)

        if self.args.save_embeddings and self.slide_embeddings is not None:
            emb_path: Path = self.eval_output_dir / "slide_embeddings.npy"
            np.save(emb_path, self.slide_embeddings.astype(np.float32, copy=False))

        return report_payload

    def _run_single_task(self, task_name: str) -> Dict[str, Any]:
        if task_name == "linear_probe":
            return self._run_linear_probe()
        if task_name == "knn_probe":
            return self._run_knn_probe()
        if task_name == "few_shot":
            return self._run_few_shot()
        if task_name == "zero_shot":
            return self._run_zero_shot()
        if task_name == "retrieval":
            return self._run_retrieval()
        if task_name == "cross_modal_retrieval":
            return self._run_cross_modal_retrieval()
        if task_name == "report_generation":
            return self._run_report_generation()
        if task_name == "survival":
            return self._run_survival()
        raise RunEvalError(f"Unsupported task: {task_name}")

    def _run_linear_probe(self) -> Dict[str, Any]:
        features, labels, split = self._classification_inputs(require_val_optional=True)

        evaluator: LinearProbeEvaluator = LinearProbeEvaluator()
        output: Dict[str, Any] = evaluator.run_linear_probe(features=features, y=labels, split=split)

        fold_scores: Optional[List[float]] = self._extract_fold_metric_scores(
            task_output=output,
            preferred_metric="balanced_accuracy",
        )
        if fold_scores:
            output["summary_fold"] = self.stats.fold_mean_std(scores=fold_scores)

        return output

    def _run_knn_probe(self) -> Dict[str, Any]:
        features, labels, _ = self._classification_inputs(require_val_optional=True)

        evaluator: KNNProbeEvaluator = KNNProbeEvaluator(
            embedding_service=self.embedding_service,
        )
        return evaluator.run_knn_probe(
            features=features,
            y=labels,
            k=int(self.cfg.evaluation.knn_probe.k),
        )

    def _run_few_shot(self) -> Dict[str, Any]:
        features, labels, split = self._classification_inputs(require_val_optional=False)

        runs: int = (
            int(self.args.few_shot_runs)
            if self.args.few_shot_runs is not None
            else int(self.cfg.evaluation.few_shot.runs)
        )

        evaluator: FewShotEvaluator = FewShotEvaluator(
            embedding_service=self.embedding_service,
            split=split,
        )
        return evaluator.run_few_shot(
            features=features,
            y=labels,
            shots=list(self.cfg.evaluation.few_shot.shots),
            runs=runs,
        )

    def _run_zero_shot(self) -> Dict[str, Any]:
        if self.multimodal_model is None:
            raise RunEvalError("Zero-shot evaluation requires a multimodal checkpoint/model.")

        features, label_values, _ = self._classification_inputs(require_val_optional=False)
        class_names: List[str] = sorted({str(item) for item in label_values.tolist()})
        if len(class_names) < 2:
            raise RunEvalError("Zero-shot evaluation requires at least two classes.")

        class_to_index: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}
        y_index: np.ndarray = np.asarray(
            [class_to_index[str(item)] for item in label_values.tolist()],
            dtype=np.int64,
        )

        prompt_texts: List[str] = [
            f"a pathology whole-slide image of {class_name}." for class_name in class_names
        ]
        class_text_emb: np.ndarray = self.embedding_service.embed_reports(prompt_texts)

        evaluator: ZeroShotEvaluator = ZeroShotEvaluator()
        output: Dict[str, Any] = evaluator.run_zero_shot(
            slide_emb=features,
            class_text_emb=class_text_emb,
            y=y_index,
        )
        output["class_names"] = class_names
        output["prompts"] = prompt_texts
        return output

    def _run_retrieval(self) -> Dict[str, Any]:
        features, labels, _ = self._classification_inputs(require_val_optional=True)
        split_idx: Dict[str, np.ndarray] = self._split_indices(self.slide_table.frame, allow_generate=True)

        db_idx: np.ndarray = self._concat_indices([split_idx["train"], split_idx["val"]])
        query_idx: np.ndarray = split_idx["test"]

        if int(db_idx.shape[0]) == 0 or int(query_idx.shape[0]) == 0:
            raise RunEvalError("Retrieval requires non-empty database and query partitions.")

        evaluator: RetrievalEvaluator = RetrievalEvaluator(
            embedding_service=self.embedding_service,
        )
        return evaluator.run_slide_retrieval(
            query=features[query_idx],
            db=features[db_idx],
            yq=labels[query_idx],
            ydb=labels[db_idx],
        )

    def _run_cross_modal_retrieval(self) -> Dict[str, Any]:
        if self.multimodal_model is None:
            raise RunEvalError("Cross-modal retrieval requires a multimodal checkpoint/model.")

        table: pd.DataFrame = self.slide_table.frame.copy()
        report_col: Optional[str] = self.slide_table.report_col
        if report_col is None:
            report_map: Dict[str, str] = self._load_report_text_map()
            table["report_text"] = table[self.slide_table.slide_id_col].map(report_map)
            report_col = "report_text"

        valid_mask: np.ndarray = table[report_col].astype(str).str.strip().to_numpy() != ""
        table = table.loc[valid_mask].reset_index(drop=True)
        if table.empty:
            raise RunEvalError("No slide-report pairs available for cross-modal retrieval.")

        slide_ids: List[str] = [str(value) for value in table[self.slide_table.slide_id_col].tolist()]
        report_texts: List[str] = [str(value) for value in table[report_col].tolist()]
        labels: np.ndarray = np.asarray(table[self.slide_table.label_col].tolist(), dtype=object)

        slide_emb: np.ndarray = self.embedding_service.embed_slides(slide_ids)
        report_emb: np.ndarray = self.embedding_service.embed_reports(report_texts)

        evaluator: RetrievalEvaluator = RetrievalEvaluator(
            embedding_service=self.embedding_service,
        )
        output: Dict[str, Any] = evaluator.run_cross_modal_retrieval(
            slide_emb=slide_emb,
            report_emb=report_emb,
            labels=labels,
        )
        output["num_pairs"] = int(table.shape[0])
        return output

    def _run_report_generation(self) -> Dict[str, Any]:
        if self.multimodal_model is None:
            raise RunEvalError("Report generation requires a multimodal checkpoint/model.")

        dataset: BaseDataset = self.registry.create_dataset(name="stage3_dataset", cfg=self.cfg.data)
        evaluator: ReportGenerationEvaluator = ReportGenerationEvaluator(
            output_dir=str(self.eval_output_dir / "report_generation"),
            bootstrap_samples=(
                int(self.args.bootstrap_n)
                if self.args.bootstrap_n is not None
                else int(self.cfg.evaluation.bootstrap_samples)
            ),
        )
        return evaluator.run_report_generation(model=self.multimodal_model, dataset=dataset)

    def _run_survival(self) -> Dict[str, Any]:
        table: pd.DataFrame = self.slide_table.frame
        if self.slide_table.time_col is None or self.slide_table.event_col is None:
            raise RunEvalError(
                "Survival evaluation requires time/event columns in metadata. "
                "Accepted names include: time/survival_time and event/status."
            )

        features: np.ndarray = self._get_slide_embeddings(table)
        time_values: np.ndarray = np.asarray(table[self.slide_table.time_col].tolist(), dtype=np.float64)
        event_values: np.ndarray = np.asarray(table[self.slide_table.event_col].tolist(), dtype=np.int64)

        folds: List[Dict[str, Any]] = self._build_survival_folds(table)
        evaluator: SurvivalEvaluator = SurvivalEvaluator(
            embedding_service=self.embedding_service,
        )
        return evaluator.run_survival(
            features=features,
            time=time_values,
            event=event_values,
            folds=folds,
        )

    def _classification_inputs(self, require_val_optional: bool) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        table: pd.DataFrame = self.slide_table.frame
        if self.slide_table.label_col is None:
            raise RunEvalError(
                "Classification-based task requires a label column. "
                "Accepted names include: label, target, class, y."
            )

        features: np.ndarray = self._get_slide_embeddings(table)
        labels: np.ndarray = np.asarray(table[self.slide_table.label_col].tolist(), dtype=object)
        split_idx: Dict[str, np.ndarray] = self._split_indices(
            frame=table,
            allow_generate=True,
        )

        split_payload: Dict[str, Any] = {
            "train": split_idx["train"].astype(np.int64, copy=False),
            "test": split_idx["test"].astype(np.int64, copy=False),
        }
        if int(split_idx["val"].shape[0]) > 0 and require_val_optional:
            split_payload["val"] = split_idx["val"].astype(np.int64, copy=False)

        return features, labels, split_payload

    def _build_statistics_block(self, results: Mapping[str, Any]) -> Dict[str, Any]:
        stats_block: Dict[str, Any] = {}

        linear_payload: Optional[Mapping[str, Any]] = (
            results.get("linear_probe") if isinstance(results.get("linear_probe"), Mapping) else None
        )
        if linear_payload is not None:
            fold_scores: Optional[List[float]] = self._extract_fold_metric_scores(
                task_output=linear_payload,
                preferred_metric="balanced_accuracy",
            )
            if fold_scores:
                stats_block["linear_probe_fold"] = self.stats.fold_mean_std(scores=fold_scores)

        if "knn_probe" in results:
            knn_metrics: Mapping[str, Any] = results["knn_probe"].get("knn", {}).get("metrics", {})
            if "balanced_accuracy" in knn_metrics:
                stats_block["knn_probe"] = {
                    "balanced_accuracy": float(knn_metrics["balanced_accuracy"]),
                }

        glmm_df: pd.DataFrame = self._build_glmm_frame(results)
        if not glmm_df.empty:
            try:
                glmm_fit: GLMMFitResult = self.stats.fit_glmm(glmm_df)
                glmm_summary: Dict[str, Any] = {
                    "backend": glmm_fit.backend,
                    "converged": bool(glmm_fit.converged),
                    "fit_error": glmm_fit.fit_error,
                    "n_rows": int(glmm_fit.prepared_df.shape[0]),
                }
                try:
                    pairwise_df: pd.DataFrame = self.stats.pairwise_tests_glmm(glmm_fit, method="tukey")
                    glmm_summary["pairwise"] = pairwise_df.to_dict(orient="records")
                except Exception as exc:  # noqa: BLE001
                    glmm_summary["pairwise_error"] = str(exc)
                stats_block["glmm"] = glmm_summary
            except Exception as exc:  # noqa: BLE001
                stats_block["glmm_error"] = str(exc)

        return stats_block

    def _build_glmm_frame(self, results: Mapping[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        if "linear_probe" in results:
            linear_payload: Mapping[str, Any] = results["linear_probe"]
            if isinstance(linear_payload.get("folds"), list):
                for fold_record in linear_payload["folds"]:
                    metrics_test: Mapping[str, Any] = fold_record.get("metrics_test", {})
                    if "balanced_accuracy" in metrics_test:
                        rows.append(
                            {
                                "method": "linear_probe",
                                "dataset": str(fold_record.get("fold_id", "fold")),
                                "score": float(metrics_test["balanced_accuracy"]),
                            }
                        )

        if "knn_probe" in results:
            knn_payload: Mapping[str, Any] = results["knn_probe"]
            knn_metrics: Mapping[str, Any] = knn_payload.get("knn", {}).get("metrics", {})
            if "balanced_accuracy" in knn_metrics:
                rows.append(
                    {
                        "method": "knn_probe",
                        "dataset": "aggregate",
                        "score": float(knn_metrics["balanced_accuracy"]),
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["method", "dataset", "score"])
        return pd.DataFrame(rows)

    def _write_task_artifact(self, task_name: str, payload: Mapping[str, Any]) -> None:
        task_dir: Path = ensure_dir(self.eval_output_dir / task_name)
        write_json(dict(payload), task_dir / f"{task_name}_metrics.json")

    def _build_models_and_load_checkpoint(
        self,
        needs_multimodal: bool,
    ) -> Tuple[TITANEncoder, Optional[CoCaModel]]:
        model_cfg: ModelConfig = self._build_model_config_from_experiment(self.cfg)

        vision_model_obj: torch.nn.Module = self.registry.create_model("titan_encoder", model_cfg)
        if not isinstance(vision_model_obj, TITANEncoder):
            raise RunEvalError(
                f"Registry returned unexpected vision model type: {type(vision_model_obj).__name__}."
            )
        vision_model: TITANEncoder = vision_model_obj

        multimodal_model: Optional[CoCaModel] = None

        if needs_multimodal:
            multimodal_obj: torch.nn.Module = self.registry.create_model("coca_multimodal", model_cfg)
            if not isinstance(multimodal_obj, CoCaModel):
                raise RunEvalError(
                    "Registry returned unexpected multimodal model type: "
                    f"{type(multimodal_obj).__name__}."
                )
            multimodal_model = multimodal_obj

            loaded_count: int = self._load_model_state_flexible(multimodal_model, self.ckpt_path)
            if loaded_count <= 0:
                raise RunEvalError("Failed to load any multimodal parameters from checkpoint.")

            for parameter in multimodal_model.parameters():
                parameter.requires_grad = False
            multimodal_model.eval()

            vision_model = multimodal_model.vision
            for parameter in vision_model.parameters():
                parameter.requires_grad = False
            vision_model.eval()

            return vision_model, multimodal_model

        vision_model.load_pretrained(str(self.ckpt_path))
        for parameter in vision_model.parameters():
            parameter.requires_grad = False
        vision_model.eval()

        return vision_model, None

    def _build_embedding_service(
        self,
        vision_model: TITANEncoder,
        multimodal_model: Optional[CoCaModel],
    ) -> EmbeddingService:
        text_model = multimodal_model.text_encoder if multimodal_model is not None else None
        return EmbeddingService(vision_model=vision_model, text_model=text_model)

    def _get_slide_embeddings(self, table: pd.DataFrame) -> np.ndarray:
        if self.slide_embeddings is not None and int(self.slide_embeddings.shape[0]) == int(table.shape[0]):
            return self.slide_embeddings

        slide_ids: List[str] = [str(value) for value in table[self.slide_table.slide_id_col].tolist()]
        embeddings: np.ndarray = self.embedding_service.embed_slides(slide_ids)

        if embeddings.ndim != 2:
            raise RunEvalError(f"Expected rank-2 slide embeddings, got shape={tuple(embeddings.shape)}.")
        if int(embeddings.shape[0]) != int(table.shape[0]):
            raise RunEvalError(
                "Slide embedding row count mismatch: "
                f"embeddings={embeddings.shape[0]}, rows={table.shape[0]}."
            )
        if int(embeddings.shape[1]) != _FEATURE_DIM:
            raise RunEvalError(
                f"Slide embedding feature dimension must be {_FEATURE_DIM}, got {embeddings.shape[1]}."
            )
        if not np.isfinite(embeddings).all():
            raise RunEvalError("Non-finite values found in slide embeddings.")

        self.slide_embeddings = embeddings.astype(np.float32, copy=False)
        return self.slide_embeddings

    def _split_indices(self, frame: pd.DataFrame, allow_generate: bool) -> Dict[str, np.ndarray]:
        if self.slide_table.split_col not in frame.columns:
            if not allow_generate:
                raise RunEvalError("Split column is missing and synthetic split generation is disabled.")
            return self._generate_default_split_indices(frame)

        split_values: pd.Series = frame[self.slide_table.split_col].astype(str).str.strip().str.lower()

        train_idx: np.ndarray = np.nonzero(split_values.to_numpy() == "train")[0].astype(np.int64)
        val_idx: np.ndarray = np.nonzero(split_values.to_numpy() == "val")[0].astype(np.int64)
        test_idx: np.ndarray = np.nonzero(split_values.to_numpy() == "test")[0].astype(np.int64)

        if int(train_idx.shape[0]) == 0 or int(test_idx.shape[0]) == 0:
            if not allow_generate:
                raise RunEvalError("Split metadata missing required train/test partitions.")
            return self._generate_default_split_indices(frame)

        return {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    def _generate_default_split_indices(self, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        n_rows: int = int(frame.shape[0])
        if n_rows < 3:
            raise RunEvalError("Need at least 3 rows to synthesize train/val/test splits.")

        rng: np.random.Generator = np.random.default_rng(int(self.cfg.runtime.seed))
        all_idx: np.ndarray = np.arange(n_rows, dtype=np.int64)
        shuffled: np.ndarray = rng.permutation(all_idx)

        n_train: int = max(1, int(round(0.7 * n_rows)))
        n_val: int = max(1, int(round(0.1 * n_rows)))
        if n_train + n_val >= n_rows:
            n_val = max(1, n_rows - n_train - 1)
        n_test: int = n_rows - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > 1:
                n_train -= 1
            else:
                n_val = max(1, n_val - 1)

        train_idx: np.ndarray = np.sort(shuffled[:n_train])
        val_idx: np.ndarray = np.sort(shuffled[n_train : n_train + n_val])
        test_idx: np.ndarray = np.sort(shuffled[n_train + n_val :])

        return {
            "train": train_idx.astype(np.int64, copy=False),
            "val": val_idx.astype(np.int64, copy=False),
            "test": test_idx.astype(np.int64, copy=False),
        }

    def _build_survival_folds(self, frame: pd.DataFrame) -> List[Dict[str, Any]]:
        n_rows: int = int(frame.shape[0])
        if n_rows < 5:
            raise RunEvalError("Survival evaluation requires at least 5 samples.")

        if self.slide_table.fold_col is not None and self.slide_table.fold_col in frame.columns:
            fold_series: pd.Series = frame[self.slide_table.fold_col].astype(str).str.strip()
            unique_folds: List[str] = sorted([value for value in fold_series.unique().tolist() if value != ""])
            if len(unique_folds) >= 2:
                folds: List[Dict[str, Any]] = []
                all_idx: np.ndarray = np.arange(n_rows, dtype=np.int64)
                for fold_id in unique_folds:
                    test_idx: np.ndarray = np.nonzero(fold_series.to_numpy() == fold_id)[0].astype(np.int64)
                    train_idx: np.ndarray = np.setdiff1d(all_idx, test_idx, assume_unique=False)
                    if int(train_idx.shape[0]) > 0 and int(test_idx.shape[0]) > 0:
                        folds.append(
                            {
                                "fold_id": str(fold_id),
                                "train": train_idx,
                                "test": test_idx,
                            }
                        )
                if folds:
                    return folds

        # Deterministic fallback to 5 folds.
        all_idx = np.arange(n_rows, dtype=np.int64)
        rng: np.random.Generator = np.random.default_rng(int(self.cfg.runtime.seed))
        shuffled: np.ndarray = rng.permutation(all_idx)

        n_folds: int = 5
        fold_chunks: List[np.ndarray] = [np.asarray(chunk, dtype=np.int64) for chunk in np.array_split(shuffled, n_folds)]

        folds = []
        for fold_idx, test_idx in enumerate(fold_chunks):
            train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
            if int(train_idx.shape[0]) == 0 or int(test_idx.shape[0]) == 0:
                continue
            folds.append(
                {
                    "fold_id": f"fold_{fold_idx}",
                    "train": np.sort(train_idx),
                    "test": np.sort(test_idx),
                }
            )

        if len(folds) == 0:
            raise RunEvalError("Failed to construct survival folds.")
        return folds

    def _load_report_text_map(self) -> Dict[str, str]:
        pairs_path: Path = Path(self.cfg.data.manifests.wsi_report_pairs_jsonl).expanduser().resolve()
        if not pairs_path.exists() or not pairs_path.is_file():
            return {}

        report_map: Dict[str, str] = {}

        suffix: str = pairs_path.suffix.lower()
        if suffix == ".jsonl":
            with pairs_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped: str = line.strip()
                    if not stripped:
                        continue
                    try:
                        row_obj: Mapping[str, Any] = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    slide_id: str = str(
                        row_obj.get("slide_id", row_obj.get("wsi_id", row_obj.get("id", "")))
                    ).strip()
                    text_value: str = str(
                        row_obj.get("text", row_obj.get("report", row_obj.get("caption", "")))
                    )
                    if slide_id and text_value.strip():
                        report_map[slide_id] = text_value
            return report_map

        if suffix == ".csv":
            frame: pd.DataFrame = pd.read_csv(pairs_path)
            slide_col: str = _find_first_present_col(
                frame,
                candidates=("slide_id", "wsi_id", "sample_id", "id"),
                required=False,
            )
            text_col: str = _find_first_present_col(
                frame,
                candidates=("text", "report", "caption"),
                required=False,
            )
            if slide_col and text_col:
                for _, row in frame.iterrows():
                    slide_id = str(row.get(slide_col, "")).strip()
                    text_val = str(row.get(text_col, ""))
                    if slide_id and text_val.strip():
                        report_map[slide_id] = text_val
        return report_map

    def _extract_fold_metric_scores(
        self,
        task_output: Mapping[str, Any],
        preferred_metric: str,
    ) -> Optional[List[float]]:
        folds_obj: Any = task_output.get("folds")
        if not isinstance(folds_obj, list):
            return None

        scores: List[float] = []
        for fold_item in folds_obj:
            if not isinstance(fold_item, Mapping):
                continue
            metrics_test: Any = fold_item.get("metrics_test")
            if not isinstance(metrics_test, Mapping):
                continue
            if preferred_metric in metrics_test:
                scores.append(float(metrics_test[preferred_metric]))

        return scores if scores else None

    def _load_slide_table(self, explicit_path: Optional[str]) -> SlideTable:
        candidate_paths: List[Path] = []
        if explicit_path is not None and explicit_path.strip():
            candidate_paths.append(Path(explicit_path).expanduser().resolve())

        candidate_paths.append(
            Path(self.cfg.paths.data_root).expanduser().resolve() / "processed" / "reports" / "prepared_slides.csv"
        )
        candidate_paths.append(Path(self.cfg.data.meta_csv).expanduser().resolve())
        candidate_paths.append(Path(self.cfg.data.manifests.wsi_manifest_csv).expanduser().resolve())

        selected_path: Optional[Path] = None
        for path in candidate_paths:
            if path.exists() and path.is_file():
                selected_path = path
                break

        if selected_path is None:
            attempted: str = ", ".join([str(path) for path in candidate_paths])
            raise RunEvalError(f"Unable to locate metadata CSV. Attempted: {attempted}")

        frame: pd.DataFrame = pd.read_csv(selected_path)
        if frame.empty:
            raise RunEvalError(f"Metadata CSV is empty: {selected_path}")

        slide_id_col: str = _find_first_present_col(
            frame,
            candidates=("slide_id", "wsi_id", "sample_id", "id"),
            required=True,
        )
        label_col: str = _find_first_present_col(
            frame,
            candidates=("label", "target", "class", "y"),
            required=False,
        )
        split_col: str = _find_first_present_col(
            frame,
            candidates=("split",),
            required=False,
        )
        fold_col: str = _find_first_present_col(
            frame,
            candidates=("fold",),
            required=False,
        )
        time_col: str = _find_first_present_col(
            frame,
            candidates=("time", "survival_time", "event_time", "duration"),
            required=False,
        )
        event_col: str = _find_first_present_col(
            frame,
            candidates=("event", "status", "event_observed", "censor"),
            required=False,
        )
        report_col: str = _find_first_present_col(
            frame,
            candidates=("report_text", "report", "text", "caption"),
            required=False,
        )

        normalized: pd.DataFrame = frame.copy()
        normalized = normalized.loc[normalized[slide_id_col].astype(str).str.strip() != ""].reset_index(drop=True)
        if normalized.empty:
            raise RunEvalError("No valid slide IDs found in metadata.")

        if not split_col:
            normalized[_DEFAULT_SPLIT_COL] = "test"
            split_col = _DEFAULT_SPLIT_COL
        else:
            normalized[split_col] = normalized[split_col].astype(str).str.strip().str.lower()

        return SlideTable(
            frame=normalized,
            slide_id_col=slide_id_col,
            label_col=(label_col if label_col else None),
            split_col=split_col,
            fold_col=(fold_col if fold_col else None),
            time_col=(time_col if time_col else None),
            event_col=(event_col if event_col else None),
            report_col=(report_col if report_col else None),
        )

    @staticmethod
    def _build_model_config_from_experiment(cfg: ExperimentConfig) -> ModelConfig:
        slide_cfg = cfg.model.slide_encoder
        return ModelConfig(
            embedding_dim=int(slide_cfg.embedding_dim),
            num_attention_layers=int(slide_cfg.num_layers),
            num_attention_heads=int(slide_cfg.num_attention_heads),
            mlp_hidden_dim=int(slide_cfg.mlp_hidden_dim),
            use_alibi_2d=bool(slide_cfg.use_alibi_2d),
            max_tokens_train=256,
            head_dim=int(slide_cfg.head_dim),
            positional_encoding=str(slide_cfg.positional_encoding),
            architecture=str(slide_cfg.architecture),
        )

    @staticmethod
    def _load_model_state_flexible(model: torch.nn.Module, ckpt_path: Path) -> int:
        checkpoint: Dict[str, Any] = load_checkpoint(path=str(ckpt_path), map_location="cpu")
        state_dict: Mapping[str, Any] = _extract_state_dict(checkpoint)

        current_state: Mapping[str, torch.Tensor] = model.state_dict()
        current_keys: set[str] = set(current_state.keys())

        cleaned: Dict[str, Any] = {}
        for raw_key, value in state_dict.items():
            if not isinstance(raw_key, str):
                continue
            candidate_keys: Tuple[str, ...] = (
                raw_key,
                _strip_prefix(raw_key, "module."),
                _strip_prefix(raw_key, "model."),
            )
            selected: Optional[str] = None
            for key in candidate_keys:
                if key in current_keys:
                    selected = key
                    break
            if selected is not None:
                cleaned[selected] = value

        if not cleaned:
            return 0

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        loaded_count: int = int(len(cleaned))
        min_required: int = max(1, int(0.1 * len(current_keys)))
        if loaded_count < min_required:
            raise RunEvalError(
                "Loaded too few parameters from checkpoint for target model. "
                f"loaded={loaded_count}, required_min={min_required}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        return loaded_count

    @staticmethod
    def _inspect_checkpoint(path: Path) -> CheckpointInfo:
        checkpoint: Dict[str, Any] = load_checkpoint(path=str(path), map_location="cpu")
        state_dict: Mapping[str, Any] = _extract_state_dict(checkpoint)

        has_multimodal_keys: bool = False
        for raw_key in state_dict.keys():
            if not isinstance(raw_key, str):
                continue
            if any(
                token in raw_key
                for token in (
                    "text_encoder",
                    "decoder",
                    "contrastive_pooler",
                    "reconstruction_pooler",
                    "logit_scale",
                )
            ):
                has_multimodal_keys = True
                break

        return CheckpointInfo(
            path=str(path),
            has_multimodal_keys=has_multimodal_keys,
            state_key_count=int(len(state_dict)),
        )

    @staticmethod
    def _requires_language_capabilities(tasks: Sequence[str]) -> bool:
        language_tasks: set[str] = {
            "zero_shot",
            "cross_modal_retrieval",
            "report_generation",
        }
        return any(task in language_tasks for task in tasks)

    @staticmethod
    def _resolve_checkpoint_path(explicit_ckpt: Optional[str], cfg: ExperimentConfig) -> Path:
        if explicit_ckpt is not None and explicit_ckpt.strip():
            path: Path = Path(explicit_ckpt).expanduser().resolve()
            if path.exists() and path.is_file():
                return path
            raise RunEvalError(f"Checkpoint does not exist: {path}")

        candidates: List[Path] = []
        if cfg.init_checkpoints.eval_from:
            candidates.append(Path(cfg.init_checkpoints.eval_from).expanduser().resolve())

        ckpt_root: Path = Path(cfg.paths.checkpoints_root).expanduser().resolve()
        candidates.extend(
            [
                ckpt_root / cfg.artifacts.stage3_checkpoint_name,
                ckpt_root / cfg.artifacts.stage2_checkpoint_name,
                ckpt_root / cfg.artifacts.stage1_checkpoint_name,
            ]
        )

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        attempted: str = ", ".join([str(item) for item in candidates])
        raise RunEvalError(
            "Checkpoint path was not provided and no default checkpoint was found. "
            f"Attempted: {attempted}"
        )

    @staticmethod
    def _resolve_config_path(requested_path: str) -> Path:
        candidate: Path = Path(requested_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate

        if str(requested_path).strip() == _DEFAULT_CONFIG_PATH:
            fallback: Path = Path(_FALLBACK_CONFIG_PATH).expanduser().resolve()
            if fallback.exists() and fallback.is_file():
                return fallback

        raise FileNotFoundError(
            "Config file not found. "
            f"requested='{requested_path}', fallback='{_FALLBACK_CONFIG_PATH}'."
        )

    def _load_and_override_config(self, args: EvalCliArgs) -> ExperimentConfig:
        resolved_cfg_path: Path = self._resolve_config_path(args.config)

        try:
            cfg: ExperimentConfig = ExperimentConfig.from_yaml(str(resolved_cfg_path))
        except (ConfigLoadError, ConfigValidationError) as exc:
            raise RunEvalError(f"Failed loading config from '{resolved_cfg_path}': {exc}") from exc

        cfg.mode = "eval"

        if args.seed is not None:
            cfg.runtime.seed = int(args.seed)

        if args.output_dir is not None:
            cfg.paths.output_root = str(Path(args.output_dir).expanduser().resolve())

        if args.bootstrap_n is not None:
            cfg.evaluation.bootstrap_samples = int(args.bootstrap_n)

        if args.few_shot_runs is not None:
            cfg.evaluation.few_shot.runs = int(args.few_shot_runs)

        try:
            cfg.validate()
        except ConfigValidationError as exc:
            raise RunEvalError(f"Configuration validation failed: {exc}") from exc

        return cfg

    def _validate_global_invariants(self) -> None:
        if int(self.cfg.data.patch_size) != _PATCH_SIZE_PX:
            raise RunEvalError(f"patch_size must be {_PATCH_SIZE_PX}, got {self.cfg.data.patch_size}.")
        if str(self.cfg.data.magnification) != _MAGNIFICATION:
            raise RunEvalError(
                f"magnification must be '{_MAGNIFICATION}', got '{self.cfg.data.magnification}'."
            )
        if int(self.cfg.data.feature_dim) != _FEATURE_DIM:
            raise RunEvalError(f"feature_dim must be {_FEATURE_DIM}, got {self.cfg.data.feature_dim}.")
        if tuple(self.cfg.data.roi_region_grid_size) != _STAGE1_REGION_GRID:
            raise RunEvalError(
                "roi_region_grid_size mismatch: "
                f"expected {_STAGE1_REGION_GRID}, got {tuple(self.cfg.data.roi_region_grid_size)}"
            )
        if tuple(self.cfg.data.stage3_wsi_crop_grid_size) != _STAGE3_CROP_GRID:
            raise RunEvalError(
                "stage3_wsi_crop_grid_size mismatch: "
                f"expected {_STAGE3_CROP_GRID}, got {tuple(self.cfg.data.stage3_wsi_crop_grid_size)}"
            )

        eval_cfg = self.cfg.evaluation
        if int(eval_cfg.linear_probe.l2_grid.count) != 45:
            raise RunEvalError("linear_probe.l2_grid.count must be 45.")
        if float(eval_cfg.linear_probe.l2_grid.min) != 1.0e-6:
            raise RunEvalError("linear_probe.l2_grid.min must be 1e-6.")
        if float(eval_cfg.linear_probe.l2_grid.max) != 10.0:
            raise RunEvalError("linear_probe.l2_grid.max must be 10.")
        if int(eval_cfg.knn_probe.k) != 20:
            raise RunEvalError("knn_probe.k must be 20.")

    @staticmethod
    def _parse_tasks(tasks_raw: str) -> List[str]:
        if not isinstance(tasks_raw, str) or not tasks_raw.strip():
            raise RunEvalError("--tasks must be a non-empty string.")

        parsed: List[str] = [item.strip().lower() for item in tasks_raw.split(",") if item.strip()]
        if not parsed:
            raise RunEvalError("No tasks were parsed from --tasks.")

        if any(item == "all" for item in parsed):
            return list(_DEFAULT_TASKS_ALL_ORDER)

        valid: set[str] = set(_DEFAULT_TASKS_ALL_ORDER)
        unknown: List[str] = [item for item in parsed if item not in valid]
        if unknown:
            raise RunEvalError(
                "Unknown task(s): "
                f"{unknown}. Supported tasks: {sorted(valid)}"
            )

        # Preserve user order but remove duplicates.
        deduped: List[str] = []
        seen: set[str] = set()
        for item in parsed:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        return deduped


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a usable state dict from common checkpoint payload formats."""
    if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], Mapping):
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
        return checkpoint["state_dict"]
    return checkpoint


def _strip_prefix(text: str, prefix: str) -> str:
    """Strip one prefix occurrence from text, if present."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def _find_first_present_col(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    required: bool,
) -> str:
    lower_to_original: Dict[str, str] = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        key: str = str(candidate).strip().lower()
        if key in lower_to_original:
            return lower_to_original[key]
    if required:
        raise RunEvalError(
            f"Required column not found. Candidates={list(candidates)}, available={list(frame.columns)}"
        )
    return ""


def _concat_indices(parts: Sequence[np.ndarray]) -> np.ndarray:
    arrays: List[np.ndarray] = [np.asarray(item, dtype=np.int64) for item in parts if int(np.asarray(item).size) > 0]
    if not arrays:
        return np.asarray([], dtype=np.int64)
    concatenated: np.ndarray = np.concatenate(arrays, axis=0)
    return np.unique(concatenated.astype(np.int64, copy=False))


def _parse_args() -> EvalCliArgs:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run TITAN/TITAN-V downstream evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to config YAML (defaults to config.yaml; falls back to eval config).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path to evaluate. If omitted, runner resolves default checkpoint candidates.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help=(
            "Comma-separated tasks from "
            "{linear_probe,knn_probe,few_shot,zero_shot,retrieval,cross_modal_retrieval,"
            "report_generation,survival,all}."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Optional slide-level metadata CSV override.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory override (default: config paths.output_root).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed override.",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=None,
        help="Optional bootstrap sample count override.",
    )
    parser.add_argument(
        "--few-shot-runs",
        type=int,
        default=None,
        help="Optional few-shot run count override.",
    )
    parser.add_argument(
        "--strict-task-fail",
        action="store_true",
        help="Fail immediately on the first task failure.",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save computed slide embeddings to outputs/eval/slide_embeddings.npy.",
    )

    namespace: argparse.Namespace = parser.parse_args()
    return EvalCliArgs(
        config=str(namespace.config),
        ckpt=(str(namespace.ckpt) if namespace.ckpt else None),
        tasks=str(namespace.tasks),
        metadata_csv=(str(namespace.metadata_csv) if namespace.metadata_csv else None),
        output_dir=(str(namespace.output_dir) if namespace.output_dir else None),
        seed=(int(namespace.seed) if namespace.seed is not None else None),
        bootstrap_n=(int(namespace.bootstrap_n) if namespace.bootstrap_n is not None else None),
        few_shot_runs=(
            int(namespace.few_shot_runs) if namespace.few_shot_runs is not None else None
        ),
        strict_task_fail=bool(namespace.strict_task_fail),
        save_embeddings=bool(namespace.save_embeddings),
    )


def main() -> None:
    args: EvalCliArgs = _parse_args()
    orchestrator: EvalOrchestrator = EvalOrchestrator(args=args)
    payload: Dict[str, Any] = orchestrator.run()

    summary_rows: List[Dict[str, Any]] = [
        {
            "task": task_name,
            "status": "completed",
        }
        for task_name in payload.get("tasks_completed", [])
    ]
    summary_rows.extend(
        [
            {
                "task": item.get("task", ""),
                "status": "failed",
                "error": item.get("error", ""),
            }
            for item in payload.get("tasks_failed", [])
        ]
    )

    summary_path: Path = ensure_dir(orchestrator.eval_output_dir) / "eval_task_summary.csv"
    write_csv(summary_rows, summary_path)


if __name__ == "__main__":
    main()
