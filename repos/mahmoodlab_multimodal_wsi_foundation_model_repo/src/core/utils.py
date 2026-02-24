"""Core reusable utilities for TITAN reproduction.

This module centralizes deterministic runtime setup, distributed utilities,
device/tensor helpers, filesystem and serialization helpers, checkpoint I/O,
and reproducibility validation against the provided configuration contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import shutil
import socket
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist


# -----------------------------------------------------------------------------
# Config-locked constants from the provided config.yaml and task contract.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_PATCH_FEATURE_DIM: int = 768
_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE1_GLOBAL_VIEWS: int = 2
_STAGE1_GLOBAL_GRID: Tuple[int, int] = (14, 14)
_STAGE1_LOCAL_VIEWS: int = 10
_STAGE1_LOCAL_GRID: Tuple[int, int] = (6, 6)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_MODEL_EMBED_DIM: int = 768
_MODEL_NUM_LAYERS: int = 6
_MODEL_NUM_HEADS: int = 12
_MODEL_HEAD_DIM: int = 64
_MODEL_MLP_DIM: int = 3072

_BOOTSTRAP_SAMPLES: int = 1000
_FEW_SHOT_RUNS: int = 50

_DEFAULT_ENCODING: str = "utf-8"
_DEFAULT_JSON_INDENT: int = 2

PathLike = Union[str, Path]
TensorOrArray = Union[torch.Tensor, np.ndarray]


class UtilsError(RuntimeError):
    """Base exception for utility-layer failures."""


class CheckpointError(UtilsError):
    """Raised when checkpoint save/load/resume fails."""


class ReproducibilityError(UtilsError):
    """Raised when config violates paper-locked reproducibility constraints."""


@dataclass(frozen=True)
class DistributedContext:
    """Distributed process context.

    Attributes:
        rank: Global rank.
        local_rank: Local rank on node.
        world_size: Number of processes.
        is_distributed: Whether DDP is initialized.
        is_main_process: True for rank zero.
        backend: Active backend (for example ``nccl`` or ``gloo``).
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_distributed: bool = False
    is_main_process: bool = True
    backend: str = "nccl"

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable mapping."""
        return asdict(self)


@dataclass(frozen=True)
class ResumeState:
    """Checkpoint resume metadata returned by :func:`resume_if_available`."""

    resumed: bool = False
    start_epoch: int = 0
    global_step: int = 0
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable mapping."""
        return asdict(self)


def seed_everything(seed: int = 42, deterministic: bool = True) -> Dict[str, Any]:
    """Seed Python, NumPy, and Torch RNGs.

    Args:
        seed: Global seed.
        deterministic: Whether to enforce deterministic CuDNN behavior.

    Returns:
        Dictionary with resolved seed settings for provenance logging.
    """
    if seed < 0:
        raise ValueError("seed must be >= 0.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    return {
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def seed_worker(worker_id: int, base_seed: int = 42) -> None:
    """Seed a DataLoader worker process.

    Args:
        worker_id: Worker id from DataLoader.
        base_seed: Base seed for deterministic worker derivation.
    """
    if worker_id < 0:
        raise ValueError("worker_id must be >= 0.")
    if base_seed < 0:
        raise ValueError("base_seed must be >= 0.")

    rank: int = int(os.environ.get("RANK", "0"))
    worker_seed: int = (int(base_seed) + int(rank) * 10_000 + int(worker_id)) % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def init_distributed(backend: str = "nccl", init_method: str = "env://") -> DistributedContext:
    """Initialize distributed context if environment indicates multi-process run.

    Args:
        backend: Torch distributed backend.
        init_method: Torch distributed init method.

    Returns:
        DistributedContext with process metadata.
    """
    rank: int = int(os.environ.get("RANK", "0"))
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank: int = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return DistributedContext(
            rank=0,
            local_rank=0,
            world_size=1,
            is_distributed=False,
            is_main_process=True,
            backend=backend,
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available.")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method=init_method)

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_distributed=True,
        is_main_process=(rank == 0),
        backend=backend,
    )


def barrier_if_distributed(ctx: Optional[DistributedContext]) -> None:
    """Synchronize all processes if distributed is active."""
    if ctx is None:
        return
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(ctx: Optional[DistributedContext]) -> None:
    """Destroy distributed process group if initialized."""
    if ctx is None:
        return
    if ctx.is_distributed and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ctx: Optional[DistributedContext] = None) -> bool:
    """Return True if running on the main process."""
    if ctx is None:
        return int(os.environ.get("RANK", "0")) == 0
    return bool(ctx.is_main_process)


def resolve_device(requested_device: str = "cuda") -> torch.device:
    """Resolve runtime device with safe fallback.

    Args:
        requested_device: Requested device string.

    Returns:
        Torch device object.
    """
    normalized: str = requested_device.strip().lower()

    if normalized.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(normalized)
        return torch.device("cpu")

    if normalized == "cpu":
        return torch.device("cpu")

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """Recursively move nested tensors to a target device.

    Args:
        batch: Nested structure of tensors/containers.
        device: Target torch device.
        non_blocking: Forwarded to ``Tensor.to``.

    Returns:
        Structure with tensors moved to ``device``.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device=device, non_blocking=non_blocking) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_to_device(item, device=device, non_blocking=non_blocking) for item in batch]
    if isinstance(batch, tuple):
        return tuple(move_to_device(item, device=device, non_blocking=non_blocking) for item in batch)
    return batch


def to_numpy(x: TensorOrArray) -> np.ndarray:
    """Convert torch tensor or numpy array to numpy array on CPU."""
    if isinstance(x, np.ndarray):
        return x
    if not isinstance(x, torch.Tensor):
        raise TypeError("to_numpy expects torch.Tensor or np.ndarray.")
    return x.detach().cpu().numpy()


def cast_if_needed(tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Optionally cast tensor to a given dtype."""
    if dtype is None:
        return tensor
    return tensor.to(dtype=dtype)


def ensure_dir(path: PathLike) -> Path:
    """Ensure a directory exists and return resolved path."""
    directory: Path = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_dir(path: PathLike) -> Path:
    """Ensure parent directory of a file path exists and return resolved path."""
    file_path: Path = Path(path).expanduser().resolve()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def prepare_run_dirs(output_dir: PathLike, stage: str) -> Dict[str, Path]:
    """Create stage-scoped output directories.

    Args:
        output_dir: Base output directory.
        stage: Stage string (for example ``stage1_titan_v``).

    Returns:
        Mapping with resolved directory paths.
    """
    root: Path = ensure_dir(Path(output_dir) / stage)
    directories: Dict[str, Path] = {
        "root": root,
        "checkpoints": ensure_dir(root / "checkpoints"),
        "logs": ensure_dir(root / "logs"),
        "metrics": ensure_dir(root / "metrics"),
        "artifacts": ensure_dir(root / "artifacts"),
    }
    return directories


def safe_symlink_or_copy(src: PathLike, dst: PathLike) -> Path:
    """Create symlink if possible, otherwise copy file.

    Args:
        src: Source file path.
        dst: Destination path.

    Returns:
        Resolved destination path.
    """
    src_path: Path = Path(src).expanduser().resolve()
    dst_path: Path = ensure_parent_dir(dst)

    if not src_path.exists():
        raise FileNotFoundError(f"Source does not exist: {src_path}")

    if dst_path.exists() or dst_path.is_symlink():
        if dst_path.is_dir() and not dst_path.is_symlink():
            raise IsADirectoryError(f"Destination is a directory: {dst_path}")
        dst_path.unlink()

    try:
        dst_path.symlink_to(src_path)
    except OSError:
        shutil.copy2(src_path, dst_path)

    return dst_path


def _to_serializable(obj: Any) -> Any:
    """Convert common scientific/python objects to JSON-serializable values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.astimezone(timezone.utc).isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        return _to_serializable(to_numpy(obj))
    if isinstance(obj, Mapping):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(v) for v in obj]
    if hasattr(obj, "model_dump") and callable(obj.model_dump):
        return _to_serializable(obj.model_dump())
    if hasattr(obj, "dict") and callable(obj.dict):
        return _to_serializable(obj.dict())
    if hasattr(obj, "__dict__"):
        return _to_serializable(vars(obj))
    return str(obj)


def _atomic_write_bytes(path: PathLike, payload: bytes) -> Path:
    """Atomically write bytes to file."""
    output_path: Path = ensure_parent_dir(path)
    fd: int
    temp_path: str
    fd, temp_path = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, str(output_path))
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    return output_path


def _atomic_write_text(path: PathLike, content: str, encoding: str = _DEFAULT_ENCODING) -> Path:
    """Atomically write text to file."""
    return _atomic_write_bytes(path=path, payload=content.encode(encoding))


def write_json(obj: Mapping[str, Any], path: PathLike, indent: int = _DEFAULT_JSON_INDENT) -> Path:
    """Write mapping to JSON atomically."""
    serializable: Any = _to_serializable(dict(obj))
    rendered: str = json.dumps(serializable, ensure_ascii=True, indent=indent, sort_keys=True)
    return _atomic_write_text(path=path, content=rendered + "\n")


def read_json(path: PathLike) -> Dict[str, Any]:
    """Read JSON mapping from file."""
    json_path: Path = Path(path).expanduser().resolve()
    if not json_path.exists() or not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding=_DEFAULT_ENCODING) as handle:
        loaded: Any = json.load(handle)

    if not isinstance(loaded, dict):
        raise TypeError(f"JSON root must be object/dict. Found: {type(loaded).__name__}")
    return loaded


def append_jsonl(records: Sequence[Mapping[str, Any]], path: PathLike) -> Path:
    """Append records to JSONL file.

    Args:
        records: Sequence of mappings to append.
        path: Destination JSONL path.

    Returns:
        Resolved file path.
    """
    jsonl_path: Path = ensure_parent_dir(path)
    with jsonl_path.open("a", encoding=_DEFAULT_ENCODING) as handle:
        for record in records:
            line: str = json.dumps(_to_serializable(dict(record)), ensure_ascii=True, sort_keys=True)
            handle.write(line)
            handle.write("\n")
    return jsonl_path


def write_csv(df_or_records: Union[pd.DataFrame, Sequence[Mapping[str, Any]]], path: PathLike) -> Path:
    """Write DataFrame or records to CSV atomically."""
    output_path: Path = ensure_parent_dir(path)

    if isinstance(df_or_records, pd.DataFrame):
        frame: pd.DataFrame = df_or_records.copy()
    else:
        frame = pd.DataFrame([dict(item) for item in df_or_records])

    csv_text: str = frame.to_csv(index=False)
    return _atomic_write_text(path=output_path, content=csv_text)


def read_csv(path: PathLike) -> pd.DataFrame:
    """Read CSV into pandas DataFrame."""
    csv_path: Path = Path(path).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    return pd.read_csv(csv_path)


def write_metrics_snapshot(
    metrics: Mapping[str, Any],
    out_dir: PathLike,
    step: Union[int, str],
    stage: str = "unknown",
    seed: int = 42,
) -> Path:
    """Persist a metrics snapshot for interrupted/resumable runs.

    Args:
        metrics: Metric mapping.
        out_dir: Output metrics directory.
        step: Global step or tag.
        stage: Stage identifier.
        seed: Run seed.

    Returns:
        Path to written JSON snapshot.
    """
    metrics_dir: Path = ensure_dir(out_dir)
    filename: str = f"metrics_step_{step}.json"
    payload: Dict[str, Any] = {
        "stage": stage,
        "seed": int(seed),
        "step": step,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": _to_serializable(dict(metrics)),
    }
    return write_json(payload, metrics_dir / filename)


def build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ema_teacher_state_dict: Optional[Mapping[str, Any]] = None,
    epoch: int = 0,
    global_step: int = 0,
    config_snapshot: Optional[Mapping[str, Any]] = None,
    runtime_metadata: Optional[Mapping[str, Any]] = None,
    extra_state: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build standardized checkpoint state mapping."""
    if epoch < 0:
        raise ValueError("epoch must be >= 0.")
    if global_step < 0:
        raise ValueError("global_step must be >= 0.")

    state: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_snapshot": _to_serializable(dict(config_snapshot or {})),
        "runtime_metadata": _to_serializable(dict(runtime_metadata or {})),
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        state["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        state["scaler_state_dict"] = scaler.state_dict()
    if ema_teacher_state_dict is not None:
        state["ema_teacher_state_dict"] = dict(ema_teacher_state_dict)
    if extra_state is not None:
        state["extra_state"] = dict(extra_state)

    return state


def save_checkpoint(state: Mapping[str, Any], path: PathLike, is_main_process_flag: bool = True) -> Optional[Path]:
    """Save checkpoint atomically.

    Args:
        state: Checkpoint mapping.
        path: Destination path.
        is_main_process_flag: Skip save when False.

    Returns:
        Resolved output path or ``None`` when skipped.
    """
    if not is_main_process_flag:
        return None

    checkpoint_path: Path = ensure_parent_dir(path)

    fd: int
    temp_path: str
    fd, temp_path = tempfile.mkstemp(prefix=f".{checkpoint_path.name}.", dir=str(checkpoint_path.parent))
    os.close(fd)

    try:
        torch.save(dict(state), temp_path)
        os.replace(temp_path, str(checkpoint_path))
    except Exception as exc:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise CheckpointError(f"Failed to save checkpoint to {checkpoint_path}") from exc

    return checkpoint_path


def load_checkpoint(path: PathLike, map_location: Union[str, torch.device] = "cpu") -> Dict[str, Any]:
    """Load checkpoint from disk.

    Args:
        path: Checkpoint path.
        map_location: Map location for ``torch.load``.

    Returns:
        Loaded checkpoint dictionary.
    """
    checkpoint_path: Path = Path(path).expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        payload: Any = torch.load(checkpoint_path, map_location=map_location)
    except Exception as exc:
        raise CheckpointError(f"Failed to load checkpoint from {checkpoint_path}") from exc

    if not isinstance(payload, dict):
        raise CheckpointError("Checkpoint payload must be a dictionary.")

    return payload


def resume_if_available(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ckpt_path: Optional[PathLike] = None,
    strict: bool = True,
) -> ResumeState:
    """Resume model/training state if checkpoint path is provided and exists.

    Args:
        model: Target model.
        optimizer: Optional optimizer.
        scheduler: Optional scheduler.
        scaler: Optional AMP scaler.
        ckpt_path: Optional checkpoint path.
        strict: Strict model state load.

    Returns:
        ResumeState with recovered epoch and global step.
    """
    if ckpt_path is None:
        return ResumeState(resumed=False)

    checkpoint_path: Path = Path(ckpt_path).expanduser().resolve()
    if not checkpoint_path.exists():
        return ResumeState(resumed=False)

    checkpoint: Dict[str, Any] = load_checkpoint(checkpoint_path, map_location="cpu")

    model_state: Optional[Mapping[str, Any]] = checkpoint.get("model_state_dict")
    if model_state is None:
        raise CheckpointError("Checkpoint missing required key: model_state_dict")

    model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint and hasattr(scheduler, "load_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if scaler is not None and "scaler_state_dict" in checkpoint and hasattr(scaler, "load_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch: int = int(checkpoint.get("epoch", 0)) + 1
    global_step: int = int(checkpoint.get("global_step", 0))

    return ResumeState(
        resumed=True,
        start_epoch=start_epoch,
        global_step=global_step,
        checkpoint_path=str(checkpoint_path),
    )


def collect_runtime_metadata() -> Dict[str, Any]:
    """Collect runtime metadata for reproducibility provenance."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "num_cuda_devices": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }


def config_hash(cfg: Any) -> str:
    """Compute stable hash for config snapshot.

    Args:
        cfg: Config-like object.

    Returns:
        SHA256 hex digest.
    """
    cfg_mapping: Dict[str, Any] = _cfg_to_mapping(cfg)
    serialized: str = json.dumps(_to_serializable(cfg_mapping), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode(_DEFAULT_ENCODING)).hexdigest()


def validate_repro_constants(cfg: Any) -> None:
    """Validate critical paper-locked constants from config.

    Args:
        cfg: Config-like object (dict, pydantic model, or object with attributes).

    Raises:
        ReproducibilityError: If any required constant differs.
    """
    mapping: Dict[str, Any] = _cfg_to_mapping(cfg)

    checks: List[Tuple[str, Any, Any]] = [
        ("data.wsi_patch_size_px", _get_nested(mapping, ("data", "wsi_patch_size_px"), _PATCH_SIZE_PX), _PATCH_SIZE_PX),
        ("data.patch_size", _get_nested(mapping, ("data", "patch_size"), _PATCH_SIZE_PX), _PATCH_SIZE_PX),
        ("data.patch_feature_dim", _get_nested(mapping, ("data", "patch_feature_dim"), _PATCH_FEATURE_DIM), _PATCH_FEATURE_DIM),
        ("data.feature_dim", _get_nested(mapping, ("data", "feature_dim"), _PATCH_FEATURE_DIM), _PATCH_FEATURE_DIM),
        ("data.magnification", _get_nested(mapping, ("data", "magnification"), _MAGNIFICATION), _MAGNIFICATION),
        ("data.roi_region_grid_size", tuple(_as_list(_get_nested(mapping, ("data", "roi_region_grid_size"), list(_STAGE1_REGION_GRID)))), _STAGE1_REGION_GRID),
        ("data.stage3_wsi_crop_grid_size", tuple(_as_list(_get_nested(mapping, ("data", "stage3_wsi_crop_grid_size"), list(_STAGE3_CROP_GRID)))), _STAGE3_CROP_GRID),
        ("model.slide_encoder.embedding_dim", _get_nested(mapping, ("model", "slide_encoder", "embedding_dim"), _MODEL_EMBED_DIM), _MODEL_EMBED_DIM),
        ("model.slide_encoder.num_layers", _get_nested(mapping, ("model", "slide_encoder", "num_layers"), _MODEL_NUM_LAYERS), _MODEL_NUM_LAYERS),
        ("model.slide_encoder.num_attention_heads", _get_nested(mapping, ("model", "slide_encoder", "num_attention_heads"), _MODEL_NUM_HEADS), _MODEL_NUM_HEADS),
        ("model.slide_encoder.head_dim", _get_nested(mapping, ("model", "slide_encoder", "head_dim"), _MODEL_HEAD_DIM), _MODEL_HEAD_DIM),
        ("model.slide_encoder.mlp_hidden_dim", _get_nested(mapping, ("model", "slide_encoder", "mlp_hidden_dim"), _MODEL_MLP_DIM), _MODEL_MLP_DIM),
        ("evaluation.bootstrap_samples", _get_nested(mapping, ("evaluation", "bootstrap_samples"), _BOOTSTRAP_SAMPLES), _BOOTSTRAP_SAMPLES),
        ("evaluation.few_shot.runs", _get_nested(mapping, ("evaluation", "few_shot", "runs"), _FEW_SHOT_RUNS), _FEW_SHOT_RUNS),
    ]

    for name, observed, expected in checks:
        if observed != expected:
            raise ReproducibilityError(f"Config mismatch for {name}: expected {expected!r}, got {observed!r}")

    stage1_views = _resolve_stage1_views(mapping)
    if stage1_views["global_views"] != _STAGE1_GLOBAL_VIEWS:
        raise ReproducibilityError(
            f"Stage1 global_views must be {_STAGE1_GLOBAL_VIEWS}, got {stage1_views['global_views']}"
        )
    if tuple(stage1_views["global_view_grid_size"]) != _STAGE1_GLOBAL_GRID:
        raise ReproducibilityError(
            f"Stage1 global_view_grid_size must be {_STAGE1_GLOBAL_GRID}, got {stage1_views['global_view_grid_size']}"
        )
    if stage1_views["local_views"] != _STAGE1_LOCAL_VIEWS:
        raise ReproducibilityError(
            f"Stage1 local_views must be {_STAGE1_LOCAL_VIEWS}, got {stage1_views['local_views']}"
        )
    if tuple(stage1_views["local_view_grid_size"]) != _STAGE1_LOCAL_GRID:
        raise ReproducibilityError(
            f"Stage1 local_view_grid_size must be {_STAGE1_LOCAL_GRID}, got {stage1_views['local_view_grid_size']}"
        )


def save_runtime_metadata(out_dir: PathLike, cfg: Any, stage: str = "unknown") -> Path:
    """Write runtime metadata file including config hash.

    Args:
        out_dir: Output directory.
        cfg: Config-like object.
        stage: Stage identifier.

    Returns:
        Path to metadata JSON file.
    """
    directory: Path = ensure_dir(out_dir)
    payload: Dict[str, Any] = {
        "stage": stage,
        "config_hash": config_hash(cfg),
        "runtime": collect_runtime_metadata(),
    }
    return write_json(payload, directory / "runtime_metadata.json")


def _cfg_to_mapping(cfg: Any) -> Dict[str, Any]:
    """Convert config-like object to plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "model_dump") and callable(cfg.model_dump):
        result: Any = cfg.model_dump()
        if isinstance(result, dict):
            return dict(result)
    if hasattr(cfg, "dict") and callable(cfg.dict):
        result = cfg.dict()
        if isinstance(result, dict):
            return dict(result)
    if hasattr(cfg, "__dict__"):
        raw: Any = vars(cfg)
        if isinstance(raw, dict):
            return dict(raw)
    raise TypeError("Unsupported config type for conversion to mapping.")


def _get_nested(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    """Safely retrieve nested mapping key path."""
    current: Any = mapping
    for key in keys:
        if isinstance(current, Mapping) and key in current:
            current = current[key]
        else:
            return default
    return current


def _as_list(value: Any) -> List[Any]:
    """Normalize value to list."""
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _resolve_stage1_views(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Resolve stage-1 view config across supported config shapes."""
    direct_train = _get_nested(mapping, ("training", "stage1_titan_v", "view_sampling"), None)
    if isinstance(direct_train, Mapping):
        return {
            "global_views": int(direct_train.get("global_views", _STAGE1_GLOBAL_VIEWS)),
            "global_view_grid_size": _as_list(direct_train.get("global_view_grid_size", list(_STAGE1_GLOBAL_GRID))),
            "local_views": int(direct_train.get("local_views", _STAGE1_LOCAL_VIEWS)),
            "local_view_grid_size": _as_list(direct_train.get("local_view_grid_size", list(_STAGE1_LOCAL_GRID))),
        }

    short_train = _get_nested(mapping, ("train", "views"), None)
    if isinstance(short_train, Mapping):
        return {
            "global_views": int(short_train.get("global_views", _STAGE1_GLOBAL_VIEWS)),
            "global_view_grid_size": _as_list(short_train.get("global_view_grid_size", list(_STAGE1_GLOBAL_GRID))),
            "local_views": int(short_train.get("local_views", _STAGE1_LOCAL_VIEWS)),
            "local_view_grid_size": _as_list(short_train.get("local_view_grid_size", list(_STAGE1_LOCAL_GRID))),
        }

    return {
        "global_views": _STAGE1_GLOBAL_VIEWS,
        "global_view_grid_size": list(_STAGE1_GLOBAL_GRID),
        "local_views": _STAGE1_LOCAL_VIEWS,
        "local_view_grid_size": list(_STAGE1_LOCAL_GRID),
    }


__all__ = [
    "UtilsError",
    "CheckpointError",
    "ReproducibilityError",
    "DistributedContext",
    "ResumeState",
    "seed_everything",
    "seed_worker",
    "init_distributed",
    "barrier_if_distributed",
    "cleanup_distributed",
    "is_main_process",
    "resolve_device",
    "move_to_device",
    "to_numpy",
    "cast_if_needed",
    "ensure_dir",
    "ensure_parent_dir",
    "prepare_run_dirs",
    "safe_symlink_or_copy",
    "write_json",
    "read_json",
    "append_jsonl",
    "write_csv",
    "read_csv",
    "write_metrics_snapshot",
    "build_checkpoint_state",
    "save_checkpoint",
    "load_checkpoint",
    "resume_if_available",
    "collect_runtime_metadata",
    "config_hash",
    "validate_repro_constants",
    "save_runtime_metadata",
]
