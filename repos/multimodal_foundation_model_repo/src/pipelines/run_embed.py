"""Embedding export pipeline for THREADS reproduction.

This module executes the design-locked embed stage:
- load manifest records
- resolve a trained THREADS checkpoint
- export slide-level embeddings
- export patient-level embeddings (union of patches across all WSIs per patient)

Public entrypoint:
- ``run_embed(cfg_or_path: Any = "configs/default.yaml") -> dict[str, Any]``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from src.core.config import ExperimentConfig
from src.core.logging_utils import (
    UnifiedLogger,
    capture_exception,
    finalize_run,
    init_unified_logger,
)
from src.data.feature_store import FeatureStore
from src.data.manifest_schema import ManifestRecord
from src.data.manifest_store import ManifestStore
from src.eval.embedding_export import EmbeddingExporter
from src.utils.io import write_json, write_parquet
from src.utils.seeding import seed_everything


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_CONFIG_PATH: str = "configs/default.yaml"
DEFAULT_STAGE: str = "embed"

DEFAULT_SEED: int = 42
DEFAULT_DETERMINISTIC: bool = True

DEFAULT_MANIFEST_PATH: str = "data/processed/manifests/pretrain_public_merged.parquet"
DEFAULT_FEATURE_STORE_ROOT: str = "data/processed/features"
DEFAULT_FEATURE_STORE_FMT: str = "hdf5"
DEFAULT_EMBEDDINGS_ROOT: str = "data/processed/embeddings"

DEFAULT_MODEL_NAME: str = "THREADS"

DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_PATIENT_AGGREGATION: str = "union of patches across all WSIs for a patient"
DEFAULT_USE_ALL_PATCHES: bool = True
DEFAULT_PATCH_SAMPLING: str = "none"
DEFAULT_TARGET_PRECISION_STR: str = "bf16"

DEFAULT_DEVICE_CPU: str = "cpu"
DEFAULT_DEVICE_CUDA: str = "cuda"

DEFAULT_SLIDE_OUTPUT_FILE: str = "threads_slide_embeddings.parquet"
DEFAULT_PATIENT_OUTPUT_FILE: str = "threads_patient_embeddings.parquet"
DEFAULT_REPORT_OUTPUT_FILE: str = "embed_record_report.parquet"
DEFAULT_SUMMARY_OUTPUT_FILE: str = "embed_summary.json"

DEFAULT_FAIL_ON_MISSING_FEATURES: bool = True
DEFAULT_FAIL_ON_EMPTY_RECORDS: bool = True
DEFAULT_FAIL_ON_EXPORT_ERROR: bool = True

DEFAULT_COHORT_FILTER: Tuple[str, ...] = tuple()
DEFAULT_TASK_FILTER: str = ""

_DEFAULT_CHECKPOINT_CANDIDATE_FILES: Tuple[str, ...] = (
    "best-rank.ckpt",
    "final.ckpt",
    "last.ckpt",
)


class EmbedPipelineError(Exception):
    """Base exception for embed pipeline failures."""


class EmbedConfigError(EmbedPipelineError):
    """Raised when embed runtime configuration is invalid."""


class EmbedRuntimeError(EmbedPipelineError):
    """Raised when embed stage execution fails."""


@dataclass(frozen=True)
class _RuntimeOptions:
    """Resolved embed runtime options."""

    manifest_path: str
    feature_store_root: str
    feature_store_fmt: str

    model_ckpt_path: str
    device: str

    output_slide_path: str
    output_patient_path: str

    cohort_filter: Tuple[str, ...]
    task_filter: str

    fail_on_missing_features: bool
    fail_on_empty_records: bool
    fail_on_export_error: bool


@dataclass(frozen=True)
class _RecordAudit:
    """Per-record embed readiness report row."""

    sample_id: str
    patient_id: str
    cohort: str
    has_features: bool
    selected: bool
    reason: str

    def to_row(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "patient_id": self.patient_id,
            "cohort": self.cohort,
            "has_features": bool(self.has_features),
            "selected": bool(self.selected),
            "reason": self.reason,
        }


class EmbedPipeline:
    """Stage runner for THREADS embedding export."""

    def __init__(self, config: ExperimentConfig, logger: UnifiedLogger) -> None:
        self._config: ExperimentConfig = self._validate_config(config)
        self._cfg_dict: Dict[str, Any] = self._config.to_dict()
        self._logger: UnifiedLogger = logger

        self._opts: _RuntimeOptions = self._resolve_runtime_options(self._cfg_dict, logger=self._logger)
        self._validate_paper_invariants(self._cfg_dict, self._opts)

        self._manifest_store: ManifestStore = ManifestStore(self._opts.manifest_path)
        self._feature_store: FeatureStore = FeatureStore(
            root_dir=self._opts.feature_store_root,
            fmt=self._opts.feature_store_fmt,
        )
        self._exporter: EmbeddingExporter = EmbeddingExporter(
            feature_store=self._feature_store,
            model_ckpt=self._opts.model_ckpt_path,
            device=self._opts.device,
        )

    def run(self) -> Dict[str, Any]:
        """Execute embed stage and return summary."""
        stage_start: float = perf_counter()

        all_records: List[ManifestRecord] = self._load_records()
        selected_records, audit_rows = self._filter_records(all_records)

        report_path: str = self._write_record_report(audit_rows)

        if len(selected_records) <= 0:
            message: str = (
                "No records available for embedding export after filtering and feature checks. "
                f"manifest_path={self._opts.manifest_path}, cohorts={list(self._opts.cohort_filter)}, "
                f"task_filter={self._opts.task_filter!r}"
            )
            if self._opts.fail_on_empty_records:
                raise EmbedConfigError(message)
            self._logger.log_event("embed_stage_noop", payload={"reason": message})
            summary_noop: Dict[str, Any] = {
                "stage": DEFAULT_STAGE,
                "run_id": self._logger.run_id,
                "model_name": DEFAULT_MODEL_NAME,
                "elapsed_sec": perf_counter() - stage_start,
                "record_count_all": len(all_records),
                "record_count_selected": 0,
                "record_report_path": report_path,
                "slide_embeddings_path": "",
                "patient_embeddings_path": "",
            }
            summary_path: Path = self._logger.paths.metrics / DEFAULT_SUMMARY_OUTPUT_FILE
            write_json(summary_noop, summary_path)
            self._logger.log_event("embed_stage_completed", payload=summary_noop)
            return summary_noop

        self._logger.log_event(
            "embed_stage_started",
            payload={
                "record_count_all": len(all_records),
                "record_count_selected": len(selected_records),
                "manifest_path": self._opts.manifest_path,
                "feature_store_root": self._opts.feature_store_root,
                "feature_store_fmt": self._opts.feature_store_fmt,
                "model_ckpt_path": self._opts.model_ckpt_path,
                "device": self._opts.device,
                "output_slide_path": self._opts.output_slide_path,
                "output_patient_path": self._opts.output_patient_path,
            },
        )

        slide_export_start: float = perf_counter()
        try:
            slide_output_path: str = self._exporter.export_slide_embeddings(
                records=selected_records,
                out_path=self._opts.output_slide_path,
            )
        except Exception as exc:  # noqa: BLE001
            if self._opts.fail_on_export_error:
                raise
            self._logger.log_event(
                "embed_slide_export_failed",
                payload={"error_type": type(exc).__name__, "error": str(exc)},
            )
            slide_output_path = ""
        slide_elapsed_sec: float = perf_counter() - slide_export_start

        patient_export_start: float = perf_counter()
        try:
            patient_output_path: str = self._exporter.export_patient_embeddings(
                records=selected_records,
                out_path=self._opts.output_patient_path,
            )
        except Exception as exc:  # noqa: BLE001
            if self._opts.fail_on_export_error:
                raise
            self._logger.log_event(
                "embed_patient_export_failed",
                payload={"error_type": type(exc).__name__, "error": str(exc)},
            )
            patient_output_path = ""
        patient_elapsed_sec: float = perf_counter() - patient_export_start

        unique_patients: int = len({record.patient_id for record in selected_records})

        summary: Dict[str, Any] = {
            "stage": DEFAULT_STAGE,
            "run_id": self._logger.run_id,
            "model_name": DEFAULT_MODEL_NAME,
            "elapsed_sec": perf_counter() - stage_start,
            "record_count_all": len(all_records),
            "record_count_selected": len(selected_records),
            "patient_count_selected": unique_patients,
            "record_report_path": report_path,
            "slide_embeddings_path": slide_output_path,
            "patient_embeddings_path": patient_output_path,
            "slide_export_elapsed_sec": slide_elapsed_sec,
            "patient_export_elapsed_sec": patient_elapsed_sec,
            "model_ckpt_path": self._opts.model_ckpt_path,
            "device": self._opts.device,
        }

        summary_path = self._logger.paths.metrics / DEFAULT_SUMMARY_OUTPUT_FILE
        write_json(summary, summary_path)

        self._logger.log_event("embed_stage_completed", payload=summary)
        return summary

    def _load_records(self) -> List[ManifestRecord]:
        records: List[ManifestRecord] = self._manifest_store.load()

        if len(self._opts.cohort_filter) > 0:
            records = self._manifest_store.filter_by_cohort(list(self._opts.cohort_filter))

        if self._opts.task_filter != "":
            records = [record for record in records if self._opts.task_filter in record.task_labels]

        records = sorted(records, key=lambda item: (str(item.patient_id), str(item.sample_id)))
        return records

    def _filter_records(self, records: Sequence[ManifestRecord]) -> Tuple[List[ManifestRecord], List[_RecordAudit]]:
        selected: List[ManifestRecord] = []
        audit_rows: List[_RecordAudit] = []

        for record in records:
            sample_id: str = str(record.sample_id)
            has_features: bool = bool(self._feature_store.exists(sample_id))
            reason: str = "selected"
            is_selected: bool = True

            if not has_features:
                reason = "missing_features"
                if self._opts.fail_on_missing_features:
                    raise EmbedRuntimeError(
                        "Required feature artifact is missing for embedding export: "
                        f"sample_id={sample_id}, feature_store_root={self._opts.feature_store_root}."
                    )
                is_selected = False

            audit_rows.append(
                _RecordAudit(
                    sample_id=str(record.sample_id),
                    patient_id=str(record.patient_id),
                    cohort=str(record.cohort),
                    has_features=has_features,
                    selected=is_selected,
                    reason=reason,
                )
            )

            if is_selected:
                selected.append(record)

        return selected, audit_rows

    def _write_record_report(self, audit_rows: Sequence[_RecordAudit]) -> str:
        report_df: pd.DataFrame = pd.DataFrame([item.to_row() for item in audit_rows])
        report_path: Path = self._logger.paths.metrics / DEFAULT_REPORT_OUTPUT_FILE
        write_parquet(report_df, report_path, sort_columns=True)
        return str(report_path)

    def _validate_paper_invariants(self, cfg: Mapping[str, Any], opts: _RuntimeOptions) -> None:
        slide_embedding_dim: int = _as_int(
            _first_present(
                cfg,
                (
                    ("model", "slide_embedding_dim"),
                    ("model_threads", "model", "slide_embedding_dim"),
                ),
                default=DEFAULT_EMBEDDING_DIM,
            ),
            default=DEFAULT_EMBEDDING_DIM,
        )
        if int(slide_embedding_dim) != DEFAULT_EMBEDDING_DIM:
            raise EmbedConfigError(
                f"model.slide_embedding_dim must be {DEFAULT_EMBEDDING_DIM}, got {slide_embedding_dim}."
            )

        use_all_patches: bool = _as_bool(
            _first_present(
                cfg,
                (("embedding_extraction", "slide_level", "use_all_patches"),),
                default=DEFAULT_USE_ALL_PATCHES,
            ),
            default=DEFAULT_USE_ALL_PATCHES,
        )
        if use_all_patches is not True:
            raise EmbedConfigError("embedding_extraction.slide_level.use_all_patches must be true.")

        patch_sampling: str = _as_str(
            _first_present(
                cfg,
                (("embedding_extraction", "slide_level", "patch_sampling"),),
                default=DEFAULT_PATCH_SAMPLING,
            ),
            default=DEFAULT_PATCH_SAMPLING,
        ).lower()
        if patch_sampling != DEFAULT_PATCH_SAMPLING:
            raise EmbedConfigError(
                f"embedding_extraction.slide_level.patch_sampling must be '{DEFAULT_PATCH_SAMPLING}', got {patch_sampling!r}."
            )

        patient_aggregation: str = _as_str(
            _first_present(
                cfg,
                (("embedding_extraction", "patient_level", "aggregation"),),
                default=DEFAULT_PATIENT_AGGREGATION,
            ),
            default=DEFAULT_PATIENT_AGGREGATION,
        )
        if patient_aggregation != DEFAULT_PATIENT_AGGREGATION:
            raise EmbedConfigError(
                "embedding_extraction.patient_level.aggregation must be "
                f"{DEFAULT_PATIENT_AGGREGATION!r}, got {patient_aggregation!r}."
            )

        target_precision: str = _as_str(
            _first_present(
                cfg,
                (("embedding_extraction", "hardware", "precision"),),
                default=DEFAULT_TARGET_PRECISION_STR,
            ),
            default=DEFAULT_TARGET_PRECISION_STR,
        ).lower()
        if target_precision != DEFAULT_TARGET_PRECISION_STR:
            raise EmbedConfigError(
                f"embedding_extraction.hardware.precision must be '{DEFAULT_TARGET_PRECISION_STR}', got {target_precision!r}."
            )

        if not Path(opts.model_ckpt_path).expanduser().resolve().is_file():
            raise EmbedConfigError(f"Resolved model checkpoint does not exist: {opts.model_ckpt_path}")

    def _resolve_runtime_options(self, cfg: Mapping[str, Any], logger: UnifiedLogger) -> _RuntimeOptions:
        manifest_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "manifest_path"),
                    ("runtime", "manifest_path"),
                    ("pretrain_public", "pretrain_public", "manifests", "files", "merged_public"),
                ),
                default=DEFAULT_MANIFEST_PATH,
            ),
            default=DEFAULT_MANIFEST_PATH,
        )

        feature_store_root: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "feature_store_root"),
                    ("runtime", "feature_store_root"),
                    ("pretrain_public", "pretrain_public", "io_roots", "features_root"),
                ),
                default=DEFAULT_FEATURE_STORE_ROOT,
            ),
            default=DEFAULT_FEATURE_STORE_ROOT,
        )

        feature_store_fmt: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "feature_store_fmt"),
                    ("runtime", "feature_store_fmt"),
                ),
                default=DEFAULT_FEATURE_STORE_FMT,
            ),
            default=DEFAULT_FEATURE_STORE_FMT,
        )

        cohort_filter: Tuple[str, ...] = _as_tuple_of_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "cohorts"),
                    ("runtime", "cohort_filter"),
                    ("runtime", "embed_cohorts"),
                ),
                default=list(DEFAULT_COHORT_FILTER),
            )
        )

        task_filter: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "task_name"),
                    ("runtime", "task_name"),
                ),
                default=DEFAULT_TASK_FILTER,
            ),
            default=DEFAULT_TASK_FILTER,
        )

        device: str = self._resolve_device(cfg)

        model_ckpt_path: str = self._resolve_checkpoint_path(cfg=cfg, logger=logger)

        default_embeddings_root: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "embeddings_root"),
                    ("runtime", "embeddings_root"),
                    ("pretrain_public", "pretrain_public", "io_roots", "embeddings_root"),
                    ("downstream_public", "downstream_public", "io", "embeddings_root"),
                ),
                default=DEFAULT_EMBEDDINGS_ROOT,
            ),
            default=DEFAULT_EMBEDDINGS_ROOT,
        )
        embeddings_root_path: Path = Path(default_embeddings_root).expanduser().resolve()

        slide_output_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "output_slide_path"),
                    ("runtime", "embed", "slide_output_path"),
                ),
                default=str(embeddings_root_path / DEFAULT_SLIDE_OUTPUT_FILE),
            ),
            default=str(embeddings_root_path / DEFAULT_SLIDE_OUTPUT_FILE),
        )
        patient_output_path: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "output_patient_path"),
                    ("runtime", "embed", "patient_output_path"),
                ),
                default=str(embeddings_root_path / DEFAULT_PATIENT_OUTPUT_FILE),
            ),
            default=str(embeddings_root_path / DEFAULT_PATIENT_OUTPUT_FILE),
        )

        fail_on_missing_features: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "fail_on_missing_features"),
                    ("runtime", "fail_on_missing_features"),
                ),
                default=DEFAULT_FAIL_ON_MISSING_FEATURES,
            ),
            default=DEFAULT_FAIL_ON_MISSING_FEATURES,
        )
        fail_on_empty_records: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "fail_on_empty_records"),
                    ("runtime", "fail_on_empty_records"),
                ),
                default=DEFAULT_FAIL_ON_EMPTY_RECORDS,
            ),
            default=DEFAULT_FAIL_ON_EMPTY_RECORDS,
        )
        fail_on_export_error: bool = _as_bool(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "fail_on_export_error"),
                    ("runtime", "fail_on_export_error"),
                ),
                default=DEFAULT_FAIL_ON_EXPORT_ERROR,
            ),
            default=DEFAULT_FAIL_ON_EXPORT_ERROR,
        )

        return _RuntimeOptions(
            manifest_path=manifest_path,
            feature_store_root=feature_store_root,
            feature_store_fmt=feature_store_fmt,
            model_ckpt_path=model_ckpt_path,
            device=device,
            output_slide_path=slide_output_path,
            output_patient_path=patient_output_path,
            cohort_filter=cohort_filter,
            task_filter=task_filter,
            fail_on_missing_features=fail_on_missing_features,
            fail_on_empty_records=fail_on_empty_records,
            fail_on_export_error=fail_on_export_error,
        )

    def _resolve_device(self, cfg: Mapping[str, Any]) -> str:
        explicit_device: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "device"),
                    ("runtime", "device"),
                ),
                default="",
            ),
            default="",
        ).lower()
        if explicit_device != "":
            return explicit_device

        requested_gpus: int = _as_int(
            _first_present(
                cfg,
                (
                    ("embedding_extraction", "hardware", "gpus"),
                    ("runtime", "embed", "gpus"),
                ),
                default=1,
            ),
            default=1,
        )

        if requested_gpus > 0:
            try:
                import torch

                if bool(torch.cuda.is_available()):
                    return DEFAULT_DEVICE_CUDA
            except Exception:
                return DEFAULT_DEVICE_CPU

        return DEFAULT_DEVICE_CPU

    def _resolve_checkpoint_path(self, cfg: Mapping[str, Any], logger: UnifiedLogger) -> str:
        explicit_ckpt: str = _as_str(
            _first_present(
                cfg,
                (
                    ("runtime", "embed", "model_ckpt"),
                    ("runtime", "embed", "checkpoint_path"),
                    ("runtime", "model_ckpt"),
                    ("runtime", "checkpoint_path"),
                    ("train_pretrain", "pretrain", "checkpointing", "resume_from"),
                ),
                default="",
            ),
            default="",
        )

        if explicit_ckpt != "":
            resolved_explicit: Path = Path(explicit_ckpt).expanduser().resolve()
            if resolved_explicit.is_file():
                return str(resolved_explicit)
            raise EmbedConfigError(
                "Configured model checkpoint path does not exist or is not a file: "
                f"{resolved_explicit}"
            )

        discovered: Optional[Path] = self._discover_default_checkpoint(cfg=cfg, logger=logger)
        if discovered is None:
            raise EmbedConfigError(
                "Unable to resolve model checkpoint for embed stage. "
                "Set runtime.embed.model_ckpt (or runtime.model_ckpt) to a valid .ckpt/.pt file."
            )
        return str(discovered)

    def _discover_default_checkpoint(self, cfg: Mapping[str, Any], logger: UnifiedLogger) -> Optional[Path]:
        candidate_paths: List[Path] = []

        run_checkpoint_dir: Path = logger.paths.checkpoints.expanduser().resolve()
        for file_name in _DEFAULT_CHECKPOINT_CANDIDATE_FILES:
            candidate_paths.append(run_checkpoint_dir / file_name)

        configured_ckpt_dir: str = _as_str(
            _first_present(
                cfg,
                (
                    ("train_pretrain", "pretrain", "checkpointing", "dirpath"),
                    ("runtime", "checkpoint_dir"),
                ),
                default="",
            ),
            default="",
        )
        if configured_ckpt_dir != "":
            configured_root: Path = Path(configured_ckpt_dir).expanduser().resolve()
            for file_name in _DEFAULT_CHECKPOINT_CANDIDATE_FILES:
                candidate_paths.append(configured_root / file_name)
            candidate_paths.extend(sorted(configured_root.glob("*.ckpt")))
            candidate_paths.extend(sorted(configured_root.glob("*.pt")))
            candidate_paths.extend(sorted(configured_root.glob("*.pth")))

        fallback_roots: Tuple[Path, ...] = (
            Path("outputs/checkpoints/pretrain").expanduser().resolve(),
            Path("outputs/checkpoints").expanduser().resolve(),
        )
        for root in fallback_roots:
            for file_name in _DEFAULT_CHECKPOINT_CANDIDATE_FILES:
                candidate_paths.append(root / file_name)

        seen: set[str] = set()
        deduped_candidates: List[Path] = []
        for item in candidate_paths:
            normalized: str = str(item)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped_candidates.append(item)

        for candidate in deduped_candidates:
            if candidate.is_file():
                return candidate

        newest_candidates: List[Tuple[float, Path]] = []
        for root in fallback_roots:
            if root.exists() and root.is_dir():
                for suffix in ("*.ckpt", "*.pt", "*.pth"):
                    for path in root.rglob(suffix):
                        if path.is_file():
                            newest_candidates.append((path.stat().st_mtime, path))

        if len(newest_candidates) > 0:
            newest_candidates.sort(key=lambda item: item[0], reverse=True)
            return newest_candidates[0][1]

        return None

    def _validate_config(self, config: ExperimentConfig) -> ExperimentConfig:
        if not isinstance(config, ExperimentConfig):
            raise EmbedConfigError(f"config must be ExperimentConfig, got {type(config).__name__}.")
        if str(config.stage).strip().lower() != DEFAULT_STAGE:
            raise EmbedConfigError(f"Expected stage='{DEFAULT_STAGE}', got '{config.stage}'.")
        return config


def run_embed(cfg_or_path: Any = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Run embed stage end-to-end.

    Args:
        cfg_or_path: Either ``ExperimentConfig`` or config YAML path.

    Returns:
        Stage summary dictionary.
    """
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

        pipeline: EmbedPipeline = EmbedPipeline(config=config, logger=logger)
        summary: Dict[str, Any] = pipeline.run()

        finalize_run(logger=logger, status="success", summary=summary)
        return summary
    except Exception as exc:  # noqa: BLE001
        capture_exception(logger=logger, exc=exc, stage_step="run_embed")
        finalize_run(
            logger=logger,
            status="failed",
            summary={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise


# -----------------------------------------------------------------------------
# Config coercion helpers
# -----------------------------------------------------------------------------
def _resolve_experiment_config(cfg_or_path: Any) -> ExperimentConfig:
    """Resolve user input to validated ExperimentConfig."""
    if isinstance(cfg_or_path, ExperimentConfig):
        cfg_or_path.validate()
        return cfg_or_path

    if isinstance(cfg_or_path, str):
        config_path: str = str(cfg_or_path).strip() or DEFAULT_CONFIG_PATH
        config: ExperimentConfig = ExperimentConfig.from_yaml(config_path)
        config.validate()
        return config

    raise EmbedConfigError(f"Unsupported cfg_or_path type: {type(cfg_or_path).__name__}.")


def _first_present(cfg: Mapping[str, Any], paths: Sequence[Tuple[str, ...]], default: Any) -> Any:
    """Return first existing value along candidate paths; otherwise default."""
    for path in paths:
        value: Any = _deep_get(cfg, path, None)
        if value is not None:
            return value
    return default


def _deep_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    """Safe nested lookup for dictionaries."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _as_str(value: Any, default: str) -> str:
    """Normalize value into non-null stripped string with fallback."""
    if value is None:
        return str(default)
    normalized: str = str(value).strip()
    if normalized == "":
        return str(default)
    return normalized


def _as_int(value: Any, default: int) -> int:
    """Normalize value into integer with fallback and strict bool rejection."""
    if value is None:
        return int(default)
    if isinstance(value, bool):
        raise EmbedConfigError("Integer value cannot be bool.")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not value.is_integer():
            raise EmbedConfigError(f"Expected integer-like value, got {value!r}.")
        return int(value)
    if isinstance(value, str):
        token: str = value.strip()
        if token == "":
            return int(default)
        try:
            return int(token)
        except ValueError as exc:
            raise EmbedConfigError(f"Expected int value, got {value!r}.") from exc
    raise EmbedConfigError(f"Expected int value, got {type(value).__name__}.")


def _as_bool(value: Any, default: bool) -> bool:
    """Normalize value into bool with permissive string parsing."""
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
    raise EmbedConfigError(f"Expected bool value, got {value!r}.")


def _as_tuple_of_str(value: Any) -> Tuple[str, ...]:
    """Normalize value into tuple[str] with empties removed."""
    if value is None:
        return tuple()

    if isinstance(value, str):
        token: str = value.strip()
        return tuple([token] if token != "" else [])

    if isinstance(value, Sequence):
        items: List[str] = []
        for item in value:
            normalized: str = str(item).strip()
            if normalized != "":
                items.append(normalized)
        return tuple(items)

    raise EmbedConfigError(f"Expected sequence of strings, got {type(value).__name__}.")


def _resolve_seed_from_cfg(cfg: Mapping[str, Any]) -> int:
    """Resolve deterministic seed from config with explicit fallback chain."""
    seed_value: Any = _first_present(
        cfg,
        (
            ("runtime", "seed"),
            ("train_pretrain", "pretrain", "training", "seed"),
            ("pretraining", "training", "seed"),
            ("downstream_public", "downstream_public", "split_policy", "seed"),
        ),
        default=DEFAULT_SEED,
    )
    return _as_int(seed_value, default=DEFAULT_SEED)


def _resolve_deterministic_from_cfg(cfg: Mapping[str, Any]) -> bool:
    """Resolve deterministic execution flag from config."""
    deterministic_value: Any = _first_present(
        cfg,
        (
            ("runtime", "deterministic"),
            ("train_pretrain", "pretrain", "training", "deterministic"),
        ),
        default=DEFAULT_DETERMINISTIC,
    )
    return _as_bool(deterministic_value, default=DEFAULT_DETERMINISTIC)


__all__ = [
    "EmbedPipelineError",
    "EmbedConfigError",
    "EmbedRuntimeError",
    "EmbedPipeline",
    "run_embed",
]
