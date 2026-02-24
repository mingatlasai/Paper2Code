"""PyTorch Lightning DataModules for THREADS reproduction.

This module provides stage-aware data loading for:
- Multimodal pretraining (WSI + RNA/DNA)
- Supervised fine-tuning and evaluation
- Embedding export

The implementation follows the provided design and `config.yaml` constraints:
- patch_size=512
- patch_stride=512
- target_magnification=20x
- patient aggregation rule: union of patches across all WSIs
- DNA input dim=1673
- pretrain batch_size_per_gpu default=300
- finetune train patch sampling=2048 (eval uses all patches)
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.feature_store import FeatureStore
from src.data.manifest_schema import ManifestRecord
from src.data.manifest_store import ManifestStore
from src.data.split_manager import SplitManager
from src.utils.seeding import build_torch_generator, seed_worker


# -----------------------------------------------------------------------------
# Constants aligned to config.yaml
# -----------------------------------------------------------------------------
DEFAULT_SEED: int = 42
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_PIN_MEMORY: bool = True
DEFAULT_PERSISTENT_WORKERS: bool = False

DEFAULT_PATCH_SIZE: int = 512
DEFAULT_PATCH_STRIDE: int = 512
DEFAULT_TARGET_MAGNIFICATION: str = "20x"

DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_DNA_INPUT_DIM: int = 1673

DEFAULT_PRETRAIN_BATCH_SIZE_PER_GPU: int = 300
DEFAULT_PRETRAIN_VAL_BATCH_SIZE: int = 64

DEFAULT_FINETUNE_BATCH_SIZE: int = 1
DEFAULT_TRAIN_PATCH_SAMPLE_COUNT: int = 2048

DEFAULT_SPLIT_DIR: str = "data/processed/splits"
DEFAULT_MANIFEST_PATH: str = "data/processed/manifests/pretrain_public_merged.parquet"
DEFAULT_FEATURE_ROOT: str = "data/processed/features"
DEFAULT_FEATURE_FMT: str = "hdf5"

DEFAULT_TASK_UNIT: str = "slide"
DEFAULT_TASK_TYPE: str = "binary_classification"
DEFAULT_TIME_KEY: str = "time"
DEFAULT_EVENT_KEY: str = "event"
DEFAULT_LABEL_KEY: str = "label"

ALLOWED_STAGES: Tuple[str, ...] = ("preprocess", "pretrain", "embed", "eval", "finetune", "fit", "validate", "test", "predict")
ALLOWED_TASK_TYPES: Tuple[str, ...] = (
    "binary_classification",
    "subtyping_multiclass",
    "grading_multiclass",
    "survival",
)
ALLOWED_TASK_UNITS: Tuple[str, ...] = ("slide", "patient")


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
class DataModuleError(Exception):
    """Base exception for datamodule failures."""


class DataModuleConfigError(DataModuleError):
    """Raised when datamodule configuration is invalid."""


class DataModuleSchemaError(DataModuleError):
    """Raised when manifest/split/schema assumptions are violated."""


class DataModuleDataError(DataModuleError):
    """Raised when data artifacts are missing/corrupt/incompatible."""


# -----------------------------------------------------------------------------
# Internal data containers
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class _PretrainItem:
    """One pretraining sample payload descriptor."""

    sample_id: str
    patient_id: str
    cohort: str
    rna_path: str
    dna_path: str


@dataclass(frozen=True)
class _SupervisedUnit:
    """One supervised task unit (slide-level or patient-level)."""

    unit_id: str
    unit_level: str
    sample_ids: Tuple[str, ...]
    patient_id: str
    label_raw: Optional[str]
    task_name: str
    task_type: str
    cohort: str
    time_value: Optional[float]
    event_value: Optional[int]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def _to_dict(cfg: Any) -> Dict[str, Any]:
    """Convert config-like object to a plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        converted: Any = cfg.to_dict()
        if isinstance(converted, Mapping):
            return dict(converted)
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            container: Any = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, Mapping):
                return dict(container)
    except Exception:
        pass
    raise DataModuleConfigError(f"Unsupported config object type: {type(cfg).__name__}.")


def _deep_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    """Safely fetch nested mapping values."""
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping):
            return default
        if key not in current:
            return default
        current = current[key]
    return current


def _first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]], default: Any = None) -> Any:
    """Return first non-None value from candidate nested paths."""
    for path in paths:
        value: Any = _deep_get(mapping, path, None)
        if value is not None:
            return value
    return default


def _normalize_stage(stage: Optional[str]) -> str:
    """Normalize runtime stage token."""
    normalized: str = "" if stage is None else str(stage).strip().lower()
    if normalized == "":
        normalized = "pretrain"
    if normalized not in ALLOWED_STAGES:
        raise DataModuleConfigError(f"Unsupported stage={stage!r}. Allowed stages: {ALLOWED_STAGES}.")
    return normalized


def _as_int(value: Any, key: str, default: int) -> int:
    """Convert config value to int with strict validation."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise DataModuleConfigError(f"{key} must be int, got bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise DataModuleConfigError(f"{key} must be integer-like, got {value}.")
        return int(value)
    if isinstance(value, str):
        value_str: str = value.strip()
        if value_str == "":
            return default
        try:
            return int(value_str)
        except ValueError as exc:
            raise DataModuleConfigError(f"{key} must be int, got {value!r}.") from exc
    raise DataModuleConfigError(f"{key} must be int, got {type(value).__name__}.")


def _as_float(value: Any, key: str, default: float) -> float:
    """Convert config value to float with strict validation."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise DataModuleConfigError(f"{key} must be float, got bool.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value_str: str = value.strip()
        if value_str == "":
            return default
        try:
            return float(value_str)
        except ValueError as exc:
            raise DataModuleConfigError(f"{key} must be float, got {value!r}.") from exc
    raise DataModuleConfigError(f"{key} must be float, got {type(value).__name__}.")


def _as_bool(value: Any, key: str, default: bool) -> bool:
    """Convert config value to bool with strict validation."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise DataModuleConfigError(f"{key} must be bool, got {value!r}.")


def _as_str(value: Any, key: str, default: str) -> str:
    """Convert config value to string."""
    if value is None:
        return default
    if isinstance(value, str):
        value_str: str = value.strip()
        return value_str if value_str != "" else default
    return str(value)


def _validate_non_empty_string(value: Any, field_name: str) -> str:
    """Ensure a value is a non-empty string."""
    normalized: str = "" if value is None else str(value).strip()
    if normalized == "":
        raise DataModuleSchemaError(f"{field_name} must be non-empty.")
    return normalized


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort finite float parser."""
    try:
        parsed: float = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _safe_int(value: Any) -> Optional[int]:
    """Best-effort integer parser."""
    try:
        parsed: int = int(value)
    except Exception:
        return None
    return parsed


def _resolve_manifest_path(cfg_dict: Mapping[str, Any], fallback: str) -> str:
    """Resolve manifest path from config with deterministic fallback."""
    return _as_str(
        _first_present(
            cfg_dict,
            paths=(
                ("pretrain_public", "pretrain_public", "manifests", "files", "merged_public"),
                ("downstream_public", "downstream_public", "io", "manifests_root"),
                ("runtime", "manifest_path"),
            ),
            default=fallback,
        ),
        key="manifest_path",
        default=fallback,
    )


def _resolve_feature_root(cfg_dict: Mapping[str, Any], fallback: str) -> str:
    """Resolve feature root path from config."""
    return _as_str(
        _first_present(
            cfg_dict,
            paths=(
                ("pretrain_public", "pretrain_public", "io_roots", "features_root"),
                ("downstream_public", "downstream_public", "io", "embeddings_root"),
                ("runtime", "feature_root"),
            ),
            default=fallback,
        ),
        key="feature_root",
        default=fallback,
    )


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    """Load JSON object from disk."""
    with path.open("r", encoding="utf-8") as file_handle:
        payload: Any = json.load(file_handle)
    if not isinstance(payload, Mapping):
        raise DataModuleDataError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


def _load_tensor_like(path_str: str) -> Any:
    """Load generic tensor-like payload from a path."""
    file_path: Path = Path(path_str).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise DataModuleDataError(f"Payload path does not exist: {file_path}")

    suffix: str = file_path.suffix.lower()
    if suffix == ".npy":
        return np.load(file_path, allow_pickle=False)
    if suffix == ".npz":
        return np.load(file_path, allow_pickle=False)
    if suffix in {".json"}:
        return _load_json_mapping(file_path)
    if suffix in {".pt", ".pth"}:
        return torch.load(file_path, map_location="cpu")

    raise DataModuleDataError(
        f"Unsupported payload suffix for path={file_path}. Supported: .npy/.npz/.json/.pt/.pth"
    )


def _extract_rna_payload(rna_path: str) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load RNA payload as (gene_ids, expr_vals, gene_mask).

    Supported formats:
    - .npz with keys among {gene_ids, expr_vals, gene_mask}
    - .npy with 1D expression values
    - .json with mapping keys above
    - .pt/.pth containing tensor or mapping
    """
    path_value: str = str(rna_path).strip()
    if path_value == "":
        return None

    payload: Any = _load_tensor_like(path_value)

    gene_ids_np: Optional[np.ndarray] = None
    expr_vals_np: Optional[np.ndarray] = None
    gene_mask_np: Optional[np.ndarray] = None

    if isinstance(payload, np.lib.npyio.NpzFile):
        keys: set[str] = set(payload.files)
        if "expr_vals" in keys:
            expr_vals_np = np.asarray(payload["expr_vals"])
        elif "expression" in keys:
            expr_vals_np = np.asarray(payload["expression"])
        if "gene_ids" in keys:
            gene_ids_np = np.asarray(payload["gene_ids"])
        if "gene_mask" in keys:
            gene_mask_np = np.asarray(payload["gene_mask"])

    elif isinstance(payload, np.ndarray):
        expr_vals_np = np.asarray(payload)

    elif isinstance(payload, Mapping):
        if "expr_vals" in payload:
            expr_vals_np = np.asarray(payload["expr_vals"])
        elif "expression" in payload:
            expr_vals_np = np.asarray(payload["expression"])
        if "gene_ids" in payload:
            gene_ids_np = np.asarray(payload["gene_ids"])
        if "gene_mask" in payload:
            gene_mask_np = np.asarray(payload["gene_mask"])

    elif isinstance(payload, torch.Tensor):
        expr_vals_np = payload.detach().cpu().numpy()

    else:
        raise DataModuleDataError(f"Unsupported RNA payload type: {type(payload).__name__}")

    if expr_vals_np is None:
        raise DataModuleDataError(f"RNA payload missing expression values: {rna_path}")

    expr_vals_np = np.asarray(expr_vals_np).reshape(-1).astype(np.float32, copy=False)
    if expr_vals_np.size == 0:
        raise DataModuleDataError(f"RNA expression vector is empty: {rna_path}")

    if gene_ids_np is None:
        gene_ids_np = np.arange(expr_vals_np.shape[0], dtype=np.int64)
    else:
        gene_ids_np = np.asarray(gene_ids_np).reshape(-1).astype(np.int64, copy=False)

    if gene_ids_np.shape[0] != expr_vals_np.shape[0]:
        raise DataModuleDataError(
            "RNA payload gene_ids and expr_vals length mismatch for "
            f"{rna_path}: {gene_ids_np.shape[0]} vs {expr_vals_np.shape[0]}"
        )

    if gene_mask_np is None:
        gene_mask_np = np.ones_like(gene_ids_np, dtype=np.bool_)
    else:
        gene_mask_np = np.asarray(gene_mask_np).reshape(-1).astype(np.bool_, copy=False)

    if gene_mask_np.shape[0] != gene_ids_np.shape[0]:
        raise DataModuleDataError(
            "RNA payload gene_mask and gene_ids length mismatch for "
            f"{rna_path}: {gene_mask_np.shape[0]} vs {gene_ids_np.shape[0]}"
        )

    gene_ids_t: torch.Tensor = torch.from_numpy(gene_ids_np).to(torch.long)
    expr_vals_t: torch.Tensor = torch.from_numpy(expr_vals_np).to(torch.float32)
    gene_mask_t: torch.Tensor = torch.from_numpy(gene_mask_np).to(torch.bool)
    return gene_ids_t, expr_vals_t, gene_mask_t


def _extract_dna_payload(dna_path: str, input_dim: int) -> Optional[torch.Tensor]:
    """Load DNA payload as fixed-length multi-hot tensor [input_dim]."""
    path_value: str = str(dna_path).strip()
    if path_value == "":
        return None

    payload: Any = _load_tensor_like(path_value)
    dna_np: Optional[np.ndarray] = None

    if isinstance(payload, np.lib.npyio.NpzFile):
        keys: set[str] = set(payload.files)
        for key in ("dna_multi_hot", "multi_hot", "values", "vector"):
            if key in keys:
                dna_np = np.asarray(payload[key])
                break
    elif isinstance(payload, np.ndarray):
        dna_np = np.asarray(payload)
    elif isinstance(payload, Mapping):
        for key in ("dna_multi_hot", "multi_hot", "values", "vector"):
            if key in payload:
                dna_np = np.asarray(payload[key])
                break
    elif isinstance(payload, torch.Tensor):
        dna_np = payload.detach().cpu().numpy()

    if dna_np is None:
        raise DataModuleDataError(f"DNA payload missing vector values: {dna_path}")

    dna_np = np.asarray(dna_np).reshape(-1).astype(np.float32, copy=False)
    if dna_np.shape[0] != int(input_dim):
        raise DataModuleDataError(
            f"DNA payload dimension mismatch for {dna_path}: expected {input_dim}, got {dna_np.shape[0]}"
        )

    return torch.from_numpy(dna_np).to(torch.float32)


def _pad_patch_features(
    feature_list: Sequence[torch.Tensor],
    *,
    feature_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length patch feature tensors to [B, N_max, F]."""
    if len(feature_list) == 0:
        raise DataModuleDataError("Cannot pad empty feature list.")

    normalized_features: List[torch.Tensor] = []
    inferred_feature_dim: Optional[int] = feature_dim

    for index, feature_tensor in enumerate(feature_list):
        if not isinstance(feature_tensor, torch.Tensor):
            raise DataModuleDataError(
                f"feature_list[{index}] must be torch.Tensor, got {type(feature_tensor).__name__}."
            )
        if feature_tensor.ndim != 2:
            raise DataModuleDataError(
                f"feature_list[{index}] must have shape [N, F], got {tuple(feature_tensor.shape)}."
            )
        if feature_tensor.shape[0] <= 0:
            raise DataModuleDataError(f"feature_list[{index}] has zero patches.")
        if inferred_feature_dim is None:
            inferred_feature_dim = int(feature_tensor.shape[1])
        if int(feature_tensor.shape[1]) != int(inferred_feature_dim):
            raise DataModuleDataError(
                "Feature dimension mismatch in batch: "
                f"expected {inferred_feature_dim}, got {feature_tensor.shape[1]}"
            )
        normalized_features.append(feature_tensor.to(torch.float32))

    if inferred_feature_dim is None:
        raise DataModuleDataError("Failed to infer feature dimension for patch padding.")

    batch_size: int = len(normalized_features)
    max_patches: int = max(int(item.shape[0]) for item in normalized_features)
    padded: torch.Tensor = torch.zeros(
        (batch_size, max_patches, inferred_feature_dim),
        dtype=torch.float32,
    )
    mask: torch.Tensor = torch.zeros((batch_size, max_patches), dtype=torch.bool)

    row_index: int
    for row_index, feature_tensor in enumerate(normalized_features):
        patch_count: int = int(feature_tensor.shape[0])
        padded[row_index, :patch_count, :] = feature_tensor
        mask[row_index, :patch_count] = True

    return padded, mask


def _pad_rna_payloads(
    rna_payloads: Sequence[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad RNA variable-length payloads and return modality mask.

    Returns:
        gene_ids: [B, G_max] int64
        expr_vals: [B, G_max] float32
        gene_mask: [B, G_max] bool
        has_rna: [B] bool
    """
    batch_size: int = len(rna_payloads)
    has_rna: torch.Tensor = torch.zeros((batch_size,), dtype=torch.bool)

    max_genes: int = 1
    payload: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    for payload in rna_payloads:
        if payload is None:
            continue
        gene_ids_tensor, _, _ = payload
        max_genes = max(max_genes, int(gene_ids_tensor.shape[0]))

    gene_ids: torch.Tensor = torch.zeros((batch_size, max_genes), dtype=torch.long)
    expr_vals: torch.Tensor = torch.zeros((batch_size, max_genes), dtype=torch.float32)
    gene_mask: torch.Tensor = torch.zeros((batch_size, max_genes), dtype=torch.bool)

    index: int
    for index, payload in enumerate(rna_payloads):
        if payload is None:
            continue
        ids_tensor, expr_tensor, mask_tensor = payload
        length: int = int(ids_tensor.shape[0])
        gene_ids[index, :length] = ids_tensor.to(torch.long)
        expr_vals[index, :length] = expr_tensor.to(torch.float32)
        gene_mask[index, :length] = mask_tensor.to(torch.bool)
        has_rna[index] = True

    return gene_ids, expr_vals, gene_mask, has_rna


def _stack_dna_payloads(
    dna_payloads: Sequence[Optional[torch.Tensor]],
    input_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack fixed-size DNA payloads and return modality mask.

    Returns:
        dna_multi_hot: [B, input_dim] float32
        has_dna: [B] bool
    """
    batch_size: int = len(dna_payloads)
    has_dna: torch.Tensor = torch.zeros((batch_size,), dtype=torch.bool)
    dna_multi_hot: torch.Tensor = torch.zeros((batch_size, int(input_dim)), dtype=torch.float32)

    index: int
    for index, payload in enumerate(dna_payloads):
        if payload is None:
            continue
        if payload.ndim != 1 or int(payload.shape[0]) != int(input_dim):
            raise DataModuleDataError(
                f"DNA payload at batch index={index} has invalid shape {tuple(payload.shape)}; "
                f"expected ({input_dim},)."
            )
        dna_multi_hot[index, :] = payload.to(torch.float32)
        has_dna[index] = True

    return dna_multi_hot, has_dna


def _sample_patch_indices(
    patch_count: int,
    max_patches: Optional[int],
    *,
    seed: int,
    sample_token: str,
) -> np.ndarray:
    """Deterministically sample patch indices without replacement."""
    if patch_count <= 0:
        raise DataModuleDataError("patch_count must be > 0.")
    if max_patches is None:
        return np.arange(patch_count, dtype=np.int64)
    if int(max_patches) <= 0:
        raise DataModuleConfigError(f"max_patches must be > 0 when provided, got {max_patches}.")

    target_count: int = min(int(max_patches), int(patch_count))
    if target_count >= patch_count:
        return np.arange(patch_count, dtype=np.int64)

    digest: str = torch.sha1(torch.tensor(list(sample_token.encode("utf-8")), dtype=torch.uint8)).numpy().tobytes().hex() if False else ""
    # Stable integer subseed from token + global seed.
    token_hash: int = int.from_bytes(sample_token.encode("utf-8"), byteorder="little", signed=False) % (2**32)
    rng: np.random.Generator = np.random.default_rng((int(seed) + int(token_hash)) % (2**32))
    chosen: np.ndarray = rng.choice(patch_count, size=target_count, replace=False)
    chosen.sort()
    return chosen.astype(np.int64, copy=False)


def _build_label_encoder(values: Sequence[Optional[str]]) -> Dict[str, int]:
    """Create deterministic string->index mapping for labels."""
    label_set: set[str] = {str(value) for value in values if value is not None and str(value).strip() != ""}
    sorted_labels: List[str] = sorted(label_set)
    return {label: index for index, label in enumerate(sorted_labels)}


def _extract_task_records(
    all_records: Sequence[ManifestRecord],
    task_name: str,
) -> List[ManifestRecord]:
    """Return records containing task label for task_name."""
    normalized_task_name: str = _validate_non_empty_string(task_name, "task_name")
    output: List[ManifestRecord] = []
    for record in all_records:
        if normalized_task_name in record.task_labels:
            output.append(record)
    return output


def _records_by_sample_id(records: Sequence[ManifestRecord]) -> Dict[str, ManifestRecord]:
    """Index records by sample_id with uniqueness checks."""
    result: Dict[str, ManifestRecord] = {}
    record: ManifestRecord
    for record in records:
        if record.sample_id in result:
            raise DataModuleSchemaError(f"Duplicate sample_id found in records: {record.sample_id}")
        result[record.sample_id] = record
    return result


def _group_records_by_patient(records: Sequence[ManifestRecord]) -> Dict[str, List[ManifestRecord]]:
    """Group records by patient_id."""
    result: Dict[str, List[ManifestRecord]] = {}
    record: ManifestRecord
    for record in records:
        result.setdefault(record.patient_id, []).append(record)
    for patient_id, patient_records in result.items():
        patient_records.sort(key=lambda item: item.sample_id)
        if patient_id.strip() == "":
            raise DataModuleSchemaError("Encountered empty patient_id while grouping records.")
    return result


# -----------------------------------------------------------------------------
# Dataset implementations
# -----------------------------------------------------------------------------
class _PretrainDataset(Dataset):
    """Dataset for THREADS multimodal pretraining."""

    def __init__(
        self,
        items: Sequence[_PretrainItem],
        *,
        feature_store: FeatureStore,
        dna_input_dim: int,
        strict_feature_checks: bool,
    ) -> None:
        self._items: List[_PretrainItem] = list(items)
        self._feature_store: FeatureStore = feature_store
        self._dna_input_dim: int = int(dna_input_dim)
        self._strict_feature_checks: bool = bool(strict_feature_checks)

        if len(self._items) == 0:
            raise DataModuleDataError("Pretrain dataset cannot be empty.")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item: _PretrainItem = self._items[int(index)]

        if not self._feature_store.exists(item.sample_id):
            if self._strict_feature_checks:
                raise DataModuleDataError(
                    f"Missing patch feature artifact for sample_id={item.sample_id}."
                )
            # Permissive mode: return empty marker, collate will filter.
            return {
                "valid": False,
                "sample_id": item.sample_id,
                "patient_id": item.patient_id,
                "cohort": item.cohort,
            }

        features_np, _coords_np = self._feature_store.read_patch_features(item.sample_id)
        feature_tensor: torch.Tensor = torch.from_numpy(np.asarray(features_np, dtype=np.float32))

        rna_payload: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = _extract_rna_payload(item.rna_path)
        dna_payload: Optional[torch.Tensor] = _extract_dna_payload(item.dna_path, input_dim=self._dna_input_dim)

        return {
            "valid": True,
            "sample_id": item.sample_id,
            "patient_id": item.patient_id,
            "cohort": item.cohort,
            "patch_features": feature_tensor,
            "rna_payload": rna_payload,
            "dna_payload": dna_payload,
        }


class _SupervisedDataset(Dataset):
    """Dataset for fine-tuning/evaluation tasks.

    Supports unit_level:
    - slide: each unit has one sample_id
    - patient: union of patches across all samples for a patient
    """

    def __init__(
        self,
        units: Sequence[_SupervisedUnit],
        *,
        feature_store: FeatureStore,
        train_mode: bool,
        train_patch_sample_count: Optional[int],
        seed: int,
        strict_feature_checks: bool,
        label_encoder: Optional[Mapping[str, int]] = None,
    ) -> None:
        self._units: List[_SupervisedUnit] = list(units)
        self._feature_store: FeatureStore = feature_store
        self._train_mode: bool = bool(train_mode)
        self._train_patch_sample_count: Optional[int] = (
            int(train_patch_sample_count) if train_patch_sample_count is not None else None
        )
        self._seed: int = int(seed)
        self._strict_feature_checks: bool = bool(strict_feature_checks)
        self._label_encoder: Dict[str, int] = dict(label_encoder) if label_encoder is not None else {}

        if len(self._units) == 0:
            raise DataModuleDataError("Supervised dataset cannot be empty.")

    def __len__(self) -> int:
        return len(self._units)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        unit: _SupervisedUnit = self._units[int(index)]

        feature_tensors: List[torch.Tensor] = []
        sample_id: str
        for sample_id in unit.sample_ids:
            if not self._feature_store.exists(sample_id):
                if self._strict_feature_checks:
                    raise DataModuleDataError(
                        f"Missing patch feature artifact for sample_id={sample_id}."
                    )
                continue
            features_np, _coords_np = self._feature_store.read_patch_features(sample_id)
            feature_tensors.append(torch.from_numpy(np.asarray(features_np, dtype=np.float32)))

        if len(feature_tensors) == 0:
            raise DataModuleDataError(
                f"No usable features found for unit_id={unit.unit_id}."
            )

        merged_features: torch.Tensor
        if len(feature_tensors) == 1:
            merged_features = feature_tensors[0]
        else:
            merged_features = torch.cat(feature_tensors, dim=0)

        if self._train_mode and self._train_patch_sample_count is not None:
            selected_indices: np.ndarray = _sample_patch_indices(
                patch_count=int(merged_features.shape[0]),
                max_patches=self._train_patch_sample_count,
                seed=self._seed,
                sample_token=f"{unit.unit_id}::{unit.task_name}",
            )
            merged_features = merged_features[torch.from_numpy(selected_indices).to(torch.long)]

        label_index: Optional[int] = None
        if unit.task_type != "survival":
            if unit.label_raw is None:
                raise DataModuleSchemaError(
                    f"Missing label for non-survival task unit_id={unit.unit_id}."
                )
            if unit.label_raw not in self._label_encoder:
                raise DataModuleSchemaError(
                    f"Label value {unit.label_raw!r} not in label encoder for task={unit.task_name}."
                )
            label_index = int(self._label_encoder[unit.label_raw])

        return {
            "unit_id": unit.unit_id,
            "unit_level": unit.unit_level,
            "sample_ids": list(unit.sample_ids),
            "patient_id": unit.patient_id,
            "task_name": unit.task_name,
            "task_type": unit.task_type,
            "cohort": unit.cohort,
            "patch_features": merged_features.to(torch.float32),
            "label_index": label_index,
            "label_raw": unit.label_raw,
            "time": unit.time_value,
            "event": unit.event_value,
        }


class _EmbeddingDataset(Dataset):
    """Dataset for deterministic embedding export."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        feature_store: FeatureStore,
        unit_level: str,
        strict_feature_checks: bool,
    ) -> None:
        normalized_unit_level: str = str(unit_level).strip().lower()
        if normalized_unit_level not in ALLOWED_TASK_UNITS:
            raise DataModuleConfigError(
                f"unit_level must be one of {ALLOWED_TASK_UNITS}, got {unit_level!r}."
            )

        self._feature_store: FeatureStore = feature_store
        self._strict_feature_checks: bool = bool(strict_feature_checks)
        self._unit_level: str = normalized_unit_level

        sorted_records: List[ManifestRecord] = sorted(
            list(records),
            key=lambda item: (item.patient_id, item.sample_id),
        )
        if len(sorted_records) == 0:
            raise DataModuleDataError("Embedding dataset cannot be empty.")

        self._records: List[ManifestRecord] = sorted_records
        self._record_index: Dict[str, ManifestRecord] = _records_by_sample_id(self._records)

        if self._unit_level == "slide":
            self._unit_ids: List[str] = [record.sample_id for record in self._records]
            self._unit_to_sample_ids: Dict[str, Tuple[str, ...]] = {
                record.sample_id: (record.sample_id,) for record in self._records
            }
            self._unit_to_patient_id: Dict[str, str] = {
                record.sample_id: record.patient_id for record in self._records
            }
            self._unit_to_cohort: Dict[str, str] = {
                record.sample_id: record.cohort for record in self._records
            }
        else:
            grouped: Dict[str, List[ManifestRecord]] = _group_records_by_patient(self._records)
            self._unit_ids = sorted(grouped.keys())
            self._unit_to_sample_ids = {
                patient_id: tuple(record.sample_id for record in patient_records)
                for patient_id, patient_records in grouped.items()
            }
            self._unit_to_patient_id = {patient_id: patient_id for patient_id in self._unit_ids}
            self._unit_to_cohort = {
                patient_id: grouped[patient_id][0].cohort for patient_id in self._unit_ids
            }

    def __len__(self) -> int:
        return len(self._unit_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        unit_id: str = self._unit_ids[int(index)]
        sample_ids: Tuple[str, ...] = self._unit_to_sample_ids[unit_id]

        feature_tensors: List[torch.Tensor] = []
        for sample_id in sample_ids:
            if not self._feature_store.exists(sample_id):
                if self._strict_feature_checks:
                    raise DataModuleDataError(
                        f"Missing patch feature artifact for sample_id={sample_id}."
                    )
                continue
            features_np, _coords_np = self._feature_store.read_patch_features(sample_id)
            feature_tensors.append(torch.from_numpy(np.asarray(features_np, dtype=np.float32)))

        if len(feature_tensors) == 0:
            raise DataModuleDataError(f"No features available for unit_id={unit_id}.")

        if len(feature_tensors) == 1:
            merged_features: torch.Tensor = feature_tensors[0]
        else:
            merged_features = torch.cat(feature_tensors, dim=0)

        return {
            "unit_id": unit_id,
            "unit_level": self._unit_level,
            "sample_ids": list(sample_ids),
            "patient_id": self._unit_to_patient_id[unit_id],
            "cohort": self._unit_to_cohort[unit_id],
            "patch_features": merged_features.to(torch.float32),
        }


# -----------------------------------------------------------------------------
# Collate functions
# -----------------------------------------------------------------------------
def _collate_pretrain(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function for pretraining batches."""
    valid_items: List[Mapping[str, Any]] = [item for item in batch if bool(item.get("valid", True))]
    if len(valid_items) == 0:
        raise DataModuleDataError("Pretrain collate received no valid items.")

    sample_ids: List[str] = [str(item["sample_id"]) for item in valid_items]
    patient_ids: List[str] = [str(item["patient_id"]) for item in valid_items]
    cohorts: List[str] = [str(item["cohort"]) for item in valid_items]

    patch_features_list: List[torch.Tensor] = [item["patch_features"] for item in valid_items]  # type: ignore[index]
    patch_features, patch_mask = _pad_patch_features(patch_features_list)

    rna_payloads: List[Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = [
        item.get("rna_payload") for item in valid_items
    ]
    gene_ids, expr_vals, gene_mask, has_rna = _pad_rna_payloads(rna_payloads)

    dna_payloads: List[Optional[torch.Tensor]] = [item.get("dna_payload") for item in valid_items]  # type: ignore[list-item]
    # Infer DNA dim from first non-null payload, else use config default.
    dna_input_dim: int = DEFAULT_DNA_INPUT_DIM
    payload: Optional[torch.Tensor]
    for payload in dna_payloads:
        if payload is not None:
            dna_input_dim = int(payload.shape[0])
            break
    dna_multi_hot, has_dna = _stack_dna_payloads(dna_payloads, input_dim=dna_input_dim)

    return {
        "sample_id": sample_ids,
        "patient_id": patient_ids,
        "cohort": cohorts,
        "patch_features": patch_features,
        "patch_mask": patch_mask,
        "has_rna": has_rna,
        "has_dna": has_dna,
        "gene_ids": gene_ids,
        "expr_vals": expr_vals,
        "gene_mask": gene_mask,
        "dna_multi_hot": dna_multi_hot,
    }


def _collate_supervised(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function for supervised/embedding batches."""
    if len(batch) == 0:
        raise DataModuleDataError("Supervised collate received empty batch.")

    patch_features_list: List[torch.Tensor] = [item["patch_features"] for item in batch]  # type: ignore[index]
    patch_features, patch_mask = _pad_patch_features(patch_features_list)

    unit_ids: List[str] = [str(item["unit_id"]) for item in batch]
    unit_levels: List[str] = [str(item["unit_level"]) for item in batch]
    sample_ids: List[List[str]] = [list(item.get("sample_ids", [])) for item in batch]
    patient_ids: List[str] = [str(item.get("patient_id", "")) for item in batch]
    cohorts: List[str] = [str(item.get("cohort", "")) for item in batch]
    task_names: List[str] = [str(item.get("task_name", "")) for item in batch]
    task_types: List[str] = [str(item.get("task_type", "")) for item in batch]

    # Optional classification labels.
    label_values: List[Optional[int]] = [
        item.get("label_index") if item.get("label_index") is not None else None for item in batch
    ]
    has_label: bool = all(label is not None for label in label_values)
    label_tensor: Optional[torch.Tensor] = None
    if has_label:
        label_tensor = torch.tensor([int(label) for label in label_values if label is not None], dtype=torch.long)

    # Optional survival fields.
    time_values: List[Optional[float]] = [
        _safe_float(item.get("time")) if item.get("time") is not None else None for item in batch
    ]
    event_values: List[Optional[int]] = [
        _safe_int(item.get("event")) if item.get("event") is not None else None for item in batch
    ]

    has_time_event: bool = all(value is not None for value in time_values) and all(value is not None for value in event_values)
    time_tensor: Optional[torch.Tensor] = None
    event_tensor: Optional[torch.Tensor] = None
    if has_time_event:
        time_tensor = torch.tensor([float(value) for value in time_values if value is not None], dtype=torch.float32)
        event_tensor = torch.tensor([int(value) for value in event_values if value is not None], dtype=torch.long)

    output: Dict[str, Any] = {
        "unit_id": unit_ids,
        "unit_level": unit_levels,
        "sample_ids": sample_ids,
        "patient_id": patient_ids,
        "cohort": cohorts,
        "task_name": task_names,
        "task_type": task_types,
        "patch_features": patch_features,
        "patch_mask": patch_mask,
    }

    if label_tensor is not None:
        output["label"] = label_tensor

    if time_tensor is not None and event_tensor is not None:
        output["time"] = time_tensor
        output["event"] = event_tensor

    return output


# -----------------------------------------------------------------------------
# Base DataModule
# -----------------------------------------------------------------------------
class _BaseThreadsDataModule(pl.LightningDataModule):
    """Common configuration and helpers for THREADS datamodules."""

    def __init__(
        self,
        cfg: Any,
        *,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        feature_root: str = DEFAULT_FEATURE_ROOT,
        feature_fmt: str = DEFAULT_FEATURE_FMT,
        split_dir: str = DEFAULT_SPLIT_DIR,
        seed: int = DEFAULT_SEED,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS,
        strict_feature_checks: bool = True,
    ) -> None:
        super().__init__()

        cfg_dict: Dict[str, Any] = _to_dict(cfg)

        self._cfg_dict: Dict[str, Any] = cfg_dict
        self._stage: str = _normalize_stage(_first_present(cfg_dict, (("runtime", "stage"),), "pretrain"))

        self._seed: int = _as_int(
            _first_present(
                cfg_dict,
                (
                    ("runtime", "seed"),
                    ("train_pretrain", "pretrain", "training", "seed"),
                    ("train_finetune", "finetune", "reproducibility", "seed"),
                ),
                seed,
            ),
            key="seed",
            default=seed,
        )

        self._manifest_path: str = _resolve_manifest_path(cfg_dict, fallback=manifest_path)
        self._feature_root: str = _resolve_feature_root(cfg_dict, fallback=feature_root)
        self._feature_fmt: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("pretrain_public", "pretrain_public", "manifests", "format"),
                    ("runtime", "feature_fmt"),
                ),
                feature_fmt,
            ),
            key="feature_fmt",
            default=feature_fmt,
        )
        self._split_dir: str = _as_str(
            _first_present(
                cfg_dict,
                (
                    ("downstream_public", "downstream_public", "io", "splits_root"),
                    ("runtime", "split_dir"),
                ),
                split_dir,
            ),
            key="split_dir",
            default=split_dir,
        )

        self._num_workers: int = _as_int(num_workers, key="num_workers", default=DEFAULT_NUM_WORKERS)
        self._pin_memory: bool = _as_bool(pin_memory, key="pin_memory", default=DEFAULT_PIN_MEMORY)
        self._persistent_workers: bool = _as_bool(
            persistent_workers,
            key="persistent_workers",
            default=DEFAULT_PERSISTENT_WORKERS,
        )
        if self._num_workers == 0:
            self._persistent_workers = False

        self._strict_feature_checks: bool = _as_bool(
            strict_feature_checks,
            key="strict_feature_checks",
            default=True,
        )

        self._manifest_store: ManifestStore = ManifestStore(self._manifest_path)
        self._feature_store: FeatureStore = FeatureStore(
            root_dir=self._feature_root,
            fmt=self._feature_fmt,
        )
        self._split_manager: SplitManager = SplitManager(split_dir=self._split_dir, seed=self._seed)

        self._all_records: List[ManifestRecord] = []

    def prepare_data(self) -> None:
        """Lightweight data checks without heavy state mutations."""
        return

    def _load_records_once(self) -> List[ManifestRecord]:
        if len(self._all_records) > 0:
            return self._all_records
        records: List[ManifestRecord] = self._manifest_store.load()
        if len(records) == 0:
            raise DataModuleDataError(f"Manifest returned no records: {self._manifest_path}")
        self._all_records = records
        return self._all_records

    def _build_loader(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        collate_fn: Any,
    ) -> DataLoader:
        """Construct deterministic DataLoader with shared defaults."""
        if batch_size <= 0:
            raise DataModuleConfigError(f"batch_size must be > 0, got {batch_size}.")

        generator: torch.Generator = build_torch_generator(seed=self._seed, device="cpu")
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self._num_workers,
            pin_memory=self._pin_memory,
            persistent_workers=self._persistent_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            collate_fn=collate_fn,
        )


# -----------------------------------------------------------------------------
# Pretraining DataModule
# -----------------------------------------------------------------------------
class ThreadsPretrainDataModule(_BaseThreadsDataModule):
    """DataModule for multimodal pretraining (WSI↔RNA/DNA)."""

    def __init__(
        self,
        cfg: Any,
        *,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        feature_root: str = DEFAULT_FEATURE_ROOT,
        feature_fmt: str = DEFAULT_FEATURE_FMT,
        split_dir: str = DEFAULT_SPLIT_DIR,
        batch_size_per_gpu: int = DEFAULT_PRETRAIN_BATCH_SIZE_PER_GPU,
        val_batch_size: int = DEFAULT_PRETRAIN_VAL_BATCH_SIZE,
        dna_input_dim: int = DEFAULT_DNA_INPUT_DIM,
        strict_feature_checks: bool = True,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS,
    ) -> None:
        super().__init__(
            cfg,
            manifest_path=manifest_path,
            feature_root=feature_root,
            feature_fmt=feature_fmt,
            split_dir=split_dir,
            strict_feature_checks=strict_feature_checks,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self._batch_size_per_gpu: int = _as_int(
            _first_present(
                self._cfg_dict,
                (
                    ("pretraining", "training", "batch_size_per_gpu"),
                    ("train_pretrain", "pretrain", "training", "batch_size_per_gpu"),
                ),
                batch_size_per_gpu,
            ),
            key="batch_size_per_gpu",
            default=batch_size_per_gpu,
        )
        self._val_batch_size: int = _as_int(val_batch_size, key="val_batch_size", default=val_batch_size)
        self._dna_input_dim: int = _as_int(
            _first_present(
                self._cfg_dict,
                (
                    ("model", "dna_encoder", "input_dim"),
                    ("model_threads", "model", "dna_encoder", "input_dim"),
                ),
                dna_input_dim,
            ),
            key="dna_input_dim",
            default=dna_input_dim,
        )

        if self._dna_input_dim != DEFAULT_DNA_INPUT_DIM:
            raise DataModuleConfigError(
                f"DNA input dim invariant violated: expected {DEFAULT_DNA_INPUT_DIM}, got {self._dna_input_dim}."
            )

        self._train_dataset: Optional[_PretrainDataset] = None
        self._val_dataset: Optional[_PretrainDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare pretraining train/val datasets."""
        _ = _normalize_stage(stage or self._stage)

        all_records: List[ManifestRecord] = self._load_records_once()

        cohort_allowlist_raw: Any = _first_present(
            self._cfg_dict,
            (("pretrain_public", "pretrain_public", "manifests", "hard_filters", "cohort_allowlist"),),
            ["TCGA", "GTEx"],
        )
        cohort_allowlist: List[str] = [str(item).strip() for item in list(cohort_allowlist_raw)]
        filtered_records: List[ManifestRecord] = [
            record for record in all_records if record.cohort in set(cohort_allowlist)
        ]

        if len(filtered_records) == 0:
            raise DataModuleDataError(
                "No pretraining records after cohort filtering. "
                f"allowlist={cohort_allowlist}"
            )

        # RNA is required in public pretraining config.
        requires_rna_pair: bool = _as_bool(
            _first_present(
                self._cfg_dict,
                (("pretrain_public", "pretrain_public", "manifests", "hard_filters", "requires_rna_pair"),),
                True,
            ),
            key="requires_rna_pair",
            default=True,
        )

        if requires_rna_pair:
            filtered_records = [record for record in filtered_records if str(record.rna_path).strip() != ""]

        if len(filtered_records) == 0:
            raise DataModuleDataError("No pretraining records with required RNA pairs.")

        # Deterministic split: prefer official split if present, else 5-fold CV on cohort labels.
        try:
            split_obj: Dict[str, Any] = self._split_manager.load_official("pretrain")
            train_ids: set[str] = set(split_obj["train_ids"])
            val_ids: set[str] = set(split_obj["test_ids"])
        except Exception:
            cv_splits: List[Dict[str, Any]] = self._split_manager.make_cv(
                records=filtered_records,
                n_folds=5,
                stratify_by="cohort",
                group_by="patient_id",
            )
            first_split: Dict[str, Any] = cv_splits[0]
            train_ids = set(first_split["train_ids"])
            val_ids = set(first_split["test_ids"])

        train_records: List[ManifestRecord] = [record for record in filtered_records if record.sample_id in train_ids]
        val_records: List[ManifestRecord] = [record for record in filtered_records if record.sample_id in val_ids]

        if len(train_records) == 0:
            raise DataModuleDataError("Pretraining train split is empty.")
        if len(val_records) == 0:
            raise DataModuleDataError("Pretraining val split is empty.")

        train_items: List[_PretrainItem] = [
            _PretrainItem(
                sample_id=record.sample_id,
                patient_id=record.patient_id,
                cohort=record.cohort,
                rna_path=record.rna_path,
                dna_path=record.dna_path,
            )
            for record in sorted(train_records, key=lambda item: item.sample_id)
        ]
        val_items: List[_PretrainItem] = [
            _PretrainItem(
                sample_id=record.sample_id,
                patient_id=record.patient_id,
                cohort=record.cohort,
                rna_path=record.rna_path,
                dna_path=record.dna_path,
            )
            for record in sorted(val_records, key=lambda item: item.sample_id)
        ]

        self._train_dataset = _PretrainDataset(
            train_items,
            feature_store=self._feature_store,
            dna_input_dim=self._dna_input_dim,
            strict_feature_checks=self._strict_feature_checks,
        )
        self._val_dataset = _PretrainDataset(
            val_items,
            feature_store=self._feature_store,
            dna_input_dim=self._dna_input_dim,
            strict_feature_checks=self._strict_feature_checks,
        )

    def train_dataloader(self) -> DataLoader:
        """Build training DataLoader."""
        if self._train_dataset is None:
            raise DataModuleDataError("setup() must be called before train_dataloader().")
        return self._build_loader(
            dataset=self._train_dataset,
            batch_size=self._batch_size_per_gpu,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate_pretrain,
        )

    def val_dataloader(self) -> DataLoader:
        """Build validation DataLoader."""
        if self._val_dataset is None:
            raise DataModuleDataError("setup() must be called before val_dataloader().")
        return self._build_loader(
            dataset=self._val_dataset,
            batch_size=self._val_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_pretrain,
        )


# -----------------------------------------------------------------------------
# Fine-tuning and Eval DataModule
# -----------------------------------------------------------------------------
class ThreadsFinetuneDataModule(_BaseThreadsDataModule):
    """DataModule for supervised fine-tuning and fold-aware evaluation."""

    def __init__(
        self,
        cfg: Any,
        *,
        task_name: str,
        task_type: str = DEFAULT_TASK_TYPE,
        unit_level: str = DEFAULT_TASK_UNIT,
        split_strategy: str = "cv5_80_20",
        fold_index: int = 0,
        k_shot: Optional[int] = None,
        train_patch_sample_count: int = DEFAULT_TRAIN_PATCH_SAMPLE_COUNT,
        batch_size: int = DEFAULT_FINETUNE_BATCH_SIZE,
        strict_feature_checks: bool = True,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        feature_root: str = DEFAULT_FEATURE_ROOT,
        feature_fmt: str = DEFAULT_FEATURE_FMT,
        split_dir: str = DEFAULT_SPLIT_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS,
        label_key: str = DEFAULT_LABEL_KEY,
        time_key: str = DEFAULT_TIME_KEY,
        event_key: str = DEFAULT_EVENT_KEY,
    ) -> None:
        super().__init__(
            cfg,
            manifest_path=manifest_path,
            feature_root=feature_root,
            feature_fmt=feature_fmt,
            split_dir=split_dir,
            strict_feature_checks=strict_feature_checks,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        normalized_task_name: str = _validate_non_empty_string(task_name, "task_name")
        normalized_task_type: str = str(task_type).strip().lower()
        normalized_unit_level: str = str(unit_level).strip().lower()
        normalized_split_strategy: str = str(split_strategy).strip().lower()

        if normalized_task_type not in ALLOWED_TASK_TYPES:
            raise DataModuleConfigError(
                f"task_type must be one of {ALLOWED_TASK_TYPES}, got {task_type!r}."
            )
        if normalized_unit_level not in ALLOWED_TASK_UNITS:
            raise DataModuleConfigError(
                f"unit_level must be one of {ALLOWED_TASK_UNITS}, got {unit_level!r}."
            )

        self._task_name: str = normalized_task_name
        self._task_type: str = normalized_task_type
        self._unit_level: str = normalized_unit_level
        self._split_strategy: str = normalized_split_strategy
        self._fold_index: int = _as_int(fold_index, key="fold_index", default=0)
        self._k_shot: Optional[int] = _as_int(k_shot, key="k_shot", default=0) if k_shot is not None else None
        self._train_patch_sample_count: int = _as_int(
            _first_present(
                self._cfg_dict,
                (
                    ("train_finetune", "finetune", "contracts", "train_patch_sampling", "patches_per_batch"),
                    ("finetuning_threads", "training", "patches_per_batch"),
                ),
                train_patch_sample_count,
            ),
            key="train_patch_sample_count",
            default=train_patch_sample_count,
        )
        self._batch_size: int = _as_int(batch_size, key="batch_size", default=batch_size)

        self._label_key: str = _as_str(label_key, key="label_key", default=DEFAULT_LABEL_KEY)
        self._time_key: str = _as_str(time_key, key="time_key", default=DEFAULT_TIME_KEY)
        self._event_key: str = _as_str(event_key, key="event_key", default=DEFAULT_EVENT_KEY)

        self._train_dataset: Optional[_SupervisedDataset] = None
        self._val_dataset: Optional[_SupervisedDataset] = None
        self._test_dataset: Optional[_SupervisedDataset] = None

        self._label_encoder: Dict[str, int] = {}
        self._active_split: Optional[Dict[str, Any]] = None

    @property
    def label_encoder(self) -> Dict[str, int]:
        """Get deterministic label encoder for non-survival tasks."""
        return dict(self._label_encoder)

    @property
    def active_split(self) -> Optional[Dict[str, Any]]:
        """Get active split object."""
        return dict(self._active_split) if self._active_split is not None else None

    def set_fold_index(self, fold_index: int) -> None:
        """Update active fold index before setup."""
        self._fold_index = _as_int(fold_index, key="fold_index", default=0)

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare train/val/test datasets for the configured task."""
        _ = _normalize_stage(stage or self._stage)

        all_records: List[ManifestRecord] = self._load_records_once()
        task_records: List[ManifestRecord] = _extract_task_records(all_records, self._task_name)

        if len(task_records) == 0:
            raise DataModuleDataError(f"No records found for task={self._task_name}.")

        split_list: List[Dict[str, Any]] = self._build_splits(task_records)
        if len(split_list) == 0:
            raise DataModuleDataError(f"No splits generated for task={self._task_name}.")

        if self._fold_index < 0 or self._fold_index >= len(split_list):
            raise DataModuleConfigError(
                f"fold_index={self._fold_index} out of range for {len(split_list)} generated splits."
            )

        selected_split: Dict[str, Any] = split_list[self._fold_index]

        if self._k_shot is not None:
            selected_split = self._split_manager.make_fewshot(selected_split, k_per_class=int(self._k_shot))

        self._active_split = dict(selected_split)

        train_ids: set[str] = set(selected_split["train_ids"])
        test_ids: set[str] = set(selected_split["test_ids"])

        record_by_sample: Dict[str, ManifestRecord] = _records_by_sample_id(task_records)
        missing_train_ids: List[str] = sorted(sample_id for sample_id in train_ids if sample_id not in record_by_sample)
        missing_test_ids: List[str] = sorted(sample_id for sample_id in test_ids if sample_id not in record_by_sample)
        if missing_train_ids or missing_test_ids:
            raise DataModuleSchemaError(
                "Split IDs missing in task records. "
                f"missing_train={missing_train_ids[:20]} missing_test={missing_test_ids[:20]}"
            )

        train_records: List[ManifestRecord] = [record_by_sample[sample_id] for sample_id in sorted(train_ids)]
        test_records: List[ManifestRecord] = [record_by_sample[sample_id] for sample_id in sorted(test_ids)]

        # Build units and label encoder (train labels only for deterministic class indexing).
        train_units: List[_SupervisedUnit] = self._build_units(train_records)
        val_units: List[_SupervisedUnit] = self._build_units(test_records)
        test_units: List[_SupervisedUnit] = self._build_units(test_records)

        if self._task_type != "survival":
            train_label_values: List[Optional[str]] = [unit.label_raw for unit in train_units]
            label_encoder: Dict[str, int] = _build_label_encoder(train_label_values)
            if len(label_encoder) < 2:
                raise DataModuleDataError(
                    f"Task={self._task_name} requires at least two classes, got {len(label_encoder)}."
                )
            self._label_encoder = label_encoder
        else:
            self._label_encoder = {}

        self._train_dataset = _SupervisedDataset(
            units=train_units,
            feature_store=self._feature_store,
            train_mode=True,
            train_patch_sample_count=self._train_patch_sample_count,
            seed=self._seed,
            strict_feature_checks=self._strict_feature_checks,
            label_encoder=self._label_encoder,
        )
        self._val_dataset = _SupervisedDataset(
            units=val_units,
            feature_store=self._feature_store,
            train_mode=False,
            train_patch_sample_count=None,
            seed=self._seed,
            strict_feature_checks=self._strict_feature_checks,
            label_encoder=self._label_encoder,
        )
        self._test_dataset = _SupervisedDataset(
            units=test_units,
            feature_store=self._feature_store,
            train_mode=False,
            train_patch_sample_count=None,
            seed=self._seed,
            strict_feature_checks=self._strict_feature_checks,
            label_encoder=self._label_encoder,
        )

    def _build_splits(self, task_records: Sequence[ManifestRecord]) -> List[Dict[str, Any]]:
        """Build task splits based on configured strategy."""
        if self._split_strategy == "official_single_fold":
            split_obj: Dict[str, Any] = self._split_manager.load_official(self._task_name)
            return [split_obj]

        if self._split_strategy == "cv5_80_20":
            return self._split_manager.make_cv(
                records=list(task_records),
                n_folds=5,
                stratify_by=self._task_name,
                group_by="patient_id",
            )

        if self._split_strategy == "mc50":
            return self._split_manager.make_monte_carlo(
                records=list(task_records),
                n_splits=50,
                test_size=0.2,
                stratify_by=self._task_name,
                group_by="patient_id",
            )

        # Attempt automatic lookup from downstream config if strategy is "auto".
        if self._split_strategy in {"auto", ""}:
            resolved_strategy: str = self._resolve_task_split_strategy_from_config()
            self._split_strategy = resolved_strategy
            return self._build_splits(task_records)

        raise DataModuleConfigError(
            "Unsupported split_strategy. Expected one of "
            "{'official_single_fold','cv5_80_20','mc50','auto'}"
            f", got {self._split_strategy!r}."
        )

    def _resolve_task_split_strategy_from_config(self) -> str:
        """Resolve split strategy from downstream task registry."""
        task_spec: Optional[Mapping[str, Any]] = self._lookup_task_spec(self._task_name)
        if task_spec is None:
            return "cv5_80_20"
        return _as_str(task_spec.get("split_strategy"), key="split_strategy", default="cv5_80_20").strip().lower()

    def _lookup_task_spec(self, task_name: str) -> Optional[Mapping[str, Any]]:
        """Find task specification in downstream_public.task_families.*.public_tasks."""
        families: Any = _deep_get(self._cfg_dict, ("downstream_public", "downstream_public", "task_families"), {})
        if not isinstance(families, Mapping):
            return None

        family_value: Any
        for family_value in families.values():
            if not isinstance(family_value, Mapping):
                continue
            public_tasks: Any = family_value.get("public_tasks", [])
            if not isinstance(public_tasks, Sequence):
                continue
            task_entry: Any
            for task_entry in public_tasks:
                if not isinstance(task_entry, Mapping):
                    continue
                if str(task_entry.get("task_name", "")).strip() == task_name:
                    return task_entry
        return None

    def _build_units(self, records: Sequence[ManifestRecord]) -> List[_SupervisedUnit]:
        """Build slide-level or patient-level task units."""
        if len(records) == 0:
            raise DataModuleDataError("Cannot build task units from empty records.")

        units: List[_SupervisedUnit] = []

        if self._unit_level == "slide":
            record: ManifestRecord
            for record in sorted(records, key=lambda item: item.sample_id):
                label_raw: Optional[str] = None
                if self._task_type != "survival":
                    label_raw = str(record.task_labels.get(self._task_name, "")).strip()
                    if label_raw == "":
                        raise DataModuleSchemaError(
                            f"Missing label for sample_id={record.sample_id} task={self._task_name}."
                        )

                time_value: Optional[float] = self._extract_time_value(record)
                event_value: Optional[int] = self._extract_event_value(record)
                if self._task_type == "survival":
                    if time_value is None or event_value is None:
                        raise DataModuleSchemaError(
                            f"Missing survival fields for sample_id={record.sample_id} task={self._task_name}."
                        )

                units.append(
                    _SupervisedUnit(
                        unit_id=record.sample_id,
                        unit_level="slide",
                        sample_ids=(record.sample_id,),
                        patient_id=record.patient_id,
                        label_raw=label_raw,
                        task_name=self._task_name,
                        task_type=self._task_type,
                        cohort=record.cohort,
                        time_value=time_value,
                        event_value=event_value,
                    )
                )
            return units

        grouped: Dict[str, List[ManifestRecord]] = _group_records_by_patient(records)
        patient_id: str
        for patient_id in sorted(grouped.keys()):
            patient_records: List[ManifestRecord] = grouped[patient_id]
            sample_ids: Tuple[str, ...] = tuple(record.sample_id for record in patient_records)

            first_record: ManifestRecord = patient_records[0]

            label_raw = None
            if self._task_type != "survival":
                patient_labels: List[str] = [
                    str(record.task_labels.get(self._task_name, "")).strip()
                    for record in patient_records
                ]
                if any(label == "" for label in patient_labels):
                    raise DataModuleSchemaError(
                        f"Missing patient-level label values for patient_id={patient_id} task={self._task_name}."
                    )
                if len(set(patient_labels)) != 1:
                    raise DataModuleSchemaError(
                        f"Inconsistent labels across patient slides for patient_id={patient_id} task={self._task_name}."
                    )
                label_raw = patient_labels[0]

            time_values: List[Optional[float]] = [self._extract_time_value(record) for record in patient_records]
            event_values: List[Optional[int]] = [self._extract_event_value(record) for record in patient_records]

            time_value: Optional[float] = None
            event_value: Optional[int] = None
            if self._task_type == "survival":
                valid_time_values: List[float] = [value for value in time_values if value is not None]
                valid_event_values: List[int] = [value for value in event_values if value is not None]
                if len(valid_time_values) == 0 or len(valid_event_values) == 0:
                    raise DataModuleSchemaError(
                        f"Missing survival fields for patient_id={patient_id} task={self._task_name}."
                    )
                # Deterministic policy: require consistency; use first value if all identical.
                if len(set(valid_time_values)) != 1 or len(set(valid_event_values)) != 1:
                    raise DataModuleSchemaError(
                        f"Inconsistent survival fields across slides for patient_id={patient_id} task={self._task_name}."
                    )
                time_value = valid_time_values[0]
                event_value = valid_event_values[0]

            units.append(
                _SupervisedUnit(
                    unit_id=patient_id,
                    unit_level="patient",
                    sample_ids=sample_ids,
                    patient_id=patient_id,
                    label_raw=label_raw,
                    task_name=self._task_name,
                    task_type=self._task_type,
                    cohort=first_record.cohort,
                    time_value=time_value,
                    event_value=event_value,
                )
            )

        return units

    def _extract_time_value(self, record: ManifestRecord) -> Optional[float]:
        """Extract survival time value from record metadata/task labels."""
        for source in (record.meta, record.task_labels):
            if self._time_key in source:
                parsed: Optional[float] = _safe_float(source[self._time_key])
                if parsed is not None:
                    return parsed
        return None

    def _extract_event_value(self, record: ManifestRecord) -> Optional[int]:
        """Extract survival event indicator from record metadata/task labels."""
        for source in (record.meta, record.task_labels):
            if self._event_key in source:
                parsed: Optional[int] = _safe_int(source[self._event_key])
                if parsed is not None:
                    return parsed
        return None

    def train_dataloader(self) -> DataLoader:
        """Build train DataLoader."""
        if self._train_dataset is None:
            raise DataModuleDataError("setup() must be called before train_dataloader().")
        return self._build_loader(
            dataset=self._train_dataset,
            batch_size=self._batch_size,
            shuffle=True,
            drop_last=False,
            collate_fn=_collate_supervised,
        )

    def val_dataloader(self) -> DataLoader:
        """Build validation DataLoader."""
        if self._val_dataset is None:
            raise DataModuleDataError("setup() must be called before val_dataloader().")
        return self._build_loader(
            dataset=self._val_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_supervised,
        )

    def test_dataloader(self) -> DataLoader:
        """Build test DataLoader."""
        if self._test_dataset is None:
            raise DataModuleDataError("setup() must be called before test_dataloader().")
        return self._build_loader(
            dataset=self._test_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_supervised,
        )

    def predict_dataloader(self) -> DataLoader:
        """Use test split for prediction by default."""
        return self.test_dataloader()


# -----------------------------------------------------------------------------
# Embedding export DataModule
# -----------------------------------------------------------------------------
class ThreadsEmbedDataModule(_BaseThreadsDataModule):
    """DataModule for deterministic slide/patient embedding export."""

    def __init__(
        self,
        cfg: Any,
        *,
        unit_level: str = "slide",
        batch_size: int = 1,
        strict_feature_checks: bool = True,
        manifest_path: str = DEFAULT_MANIFEST_PATH,
        feature_root: str = DEFAULT_FEATURE_ROOT,
        feature_fmt: str = DEFAULT_FEATURE_FMT,
        split_dir: str = DEFAULT_SPLIT_DIR,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS,
        cohorts: Optional[Sequence[str]] = None,
        task_name: Optional[str] = None,
    ) -> None:
        super().__init__(
            cfg,
            manifest_path=manifest_path,
            feature_root=feature_root,
            feature_fmt=feature_fmt,
            split_dir=split_dir,
            strict_feature_checks=strict_feature_checks,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        normalized_unit_level: str = str(unit_level).strip().lower()
        if normalized_unit_level not in ALLOWED_TASK_UNITS:
            raise DataModuleConfigError(
                f"unit_level must be one of {ALLOWED_TASK_UNITS}, got {unit_level!r}."
            )

        self._unit_level: str = normalized_unit_level
        self._batch_size: int = _as_int(batch_size, key="batch_size", default=batch_size)

        self._cohorts: Optional[List[str]] = [str(item).strip() for item in cohorts] if cohorts is not None else None
        self._task_name: Optional[str] = str(task_name).strip() if task_name is not None else None

        self._dataset: Optional[_EmbeddingDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Prepare embedding export dataset."""
        _ = _normalize_stage(stage or self._stage)

        records: List[ManifestRecord] = self._load_records_once()

        if self._cohorts is not None:
            cohort_set: set[str] = {cohort for cohort in self._cohorts if cohort != ""}
            records = [record for record in records if record.cohort in cohort_set]

        if self._task_name is not None and self._task_name != "":
            records = [record for record in records if self._task_name in record.task_labels]

        if len(records) == 0:
            raise DataModuleDataError("No records available for embedding export after filtering.")

        self._dataset = _EmbeddingDataset(
            records=records,
            feature_store=self._feature_store,
            unit_level=self._unit_level,
            strict_feature_checks=self._strict_feature_checks,
        )

    def predict_dataloader(self) -> DataLoader:
        """Build deterministic export DataLoader."""
        if self._dataset is None:
            raise DataModuleDataError("setup() must be called before predict_dataloader().")
        return self._build_loader(
            dataset=self._dataset,
            batch_size=self._batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=_collate_supervised,
        )


# -----------------------------------------------------------------------------
# Eval convenience alias (same behavior as finetune data module)
# -----------------------------------------------------------------------------
class ThreadsEvalDataModule(ThreadsFinetuneDataModule):
    """Evaluation datamodule alias.

    Uses the same fold-aware dataset construction as fine-tuning but is typically
    consumed by evaluation pipelines for deterministic split materialization.
    """


# Backward-friendly aliases matching concise naming styles.
PretrainDataModule = ThreadsPretrainDataModule
FinetuneDataModule = ThreadsFinetuneDataModule
EmbedDataModule = ThreadsEmbedDataModule
EvalDataModule = ThreadsEvalDataModule


__all__ = [
    "DataModuleError",
    "DataModuleConfigError",
    "DataModuleSchemaError",
    "DataModuleDataError",
    "ThreadsPretrainDataModule",
    "ThreadsFinetuneDataModule",
    "ThreadsEmbedDataModule",
    "ThreadsEvalDataModule",
    "PretrainDataModule",
    "FinetuneDataModule",
    "EmbedDataModule",
    "EvalDataModule",
]
