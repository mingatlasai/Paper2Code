"""Deterministic seeding utilities for THREADS reproduction.

This module centralizes seed handling across Python, NumPy, Torch, and DDP
contexts. It is designed to be called at process start by `main.py` and each
stage pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
import random
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch


SEED_CONTRACT_VERSION: str = "threads_seed_v1"
DEFAULT_GLOBAL_SEED: int = 42
DEFAULT_STAGE: str = "preprocess"
DEFAULT_DETERMINISTIC: bool = True

_ALLOWED_STAGES: tuple[str, ...] = ("preprocess", "pretrain", "embed", "eval")

# Internal process-local seeding state used by worker init and provenance.
_LAST_SEED_STATE: Optional["SeedState"] = None


class SeedingError(Exception):
    """Base exception for seeding-related failures."""


class SeedConfigurationError(SeedingError):
    """Raised when seed configuration values are invalid."""


@dataclass(frozen=True)
class SeedConfig:
    """Resolved seed configuration.

    Attributes:
        global_seed: Root seed for the run.
        stage: Runtime stage name.
        deterministic: Whether to enable deterministic backend behavior.
        strict_config_validation: Whether to fail on conflicting backend flags.
        cudnn_benchmark: CuDNN benchmark mode.
        cudnn_deterministic: CuDNN deterministic mode.
        allow_tf32: TF32 allowance for CUDA matmul/CuDNN.
    """

    global_seed: int = DEFAULT_GLOBAL_SEED
    stage: str = DEFAULT_STAGE
    deterministic: bool = DEFAULT_DETERMINISTIC
    strict_config_validation: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    allow_tf32: bool = False


@dataclass(frozen=True)
class SeedState:
    """Materialized seed state for logging and provenance."""

    root_seed: int
    stage: str
    stage_seed: int
    global_rank: int
    world_size: int
    deterministic_mode: bool
    strict_config_validation: bool
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    allow_tf32: bool
    worker_seed_scheme: str
    seed_contract_version: str = SEED_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable metadata dictionary."""
        return asdict(self)


def seed_everything(global_seed: int, deterministic: bool, stage: str) -> Dict[str, Any]:
    """Seed Python/NumPy/Torch RNGs with DDP-aware stage derivation.

    Args:
        global_seed: Root reproducibility seed.
        deterministic: Whether deterministic backend mode should be enabled.
        stage: Runtime stage (`preprocess`, `pretrain`, `embed`, or `eval`).

    Returns:
        Seed provenance dictionary for logging.

    Raises:
        SeedConfigurationError: If the arguments are invalid.
    """
    normalized_stage: str = _normalize_stage(stage)
    validated_seed: int = _validate_seed_int(global_seed)

    config: SeedConfig = SeedConfig(
        global_seed=validated_seed,
        stage=normalized_stage,
        deterministic=bool(deterministic),
        strict_config_validation=True,
        cudnn_benchmark=False,
        cudnn_deterministic=bool(deterministic),
        allow_tf32=False,
    )
    return _apply_seed_config(config).to_dict()


def derive_rank_seed(global_seed: int, stage: str, global_rank: int) -> int:
    """Derive deterministic stage/rank-specific seed.

    Args:
        global_seed: Root seed.
        stage: Runtime stage.
        global_rank: Distributed global rank.

    Returns:
        Derived 32-bit integer seed.

    Raises:
        SeedConfigurationError: If inputs are invalid.
    """
    validated_seed: int = _validate_seed_int(global_seed)
    normalized_stage: str = _normalize_stage(stage)
    validated_rank: int = _validate_non_negative_int(global_rank, key="global_rank")

    stage_hash_payload: bytes = (
        f"{SEED_CONTRACT_VERSION}:{normalized_stage}".encode("utf-8")
    )
    stage_hash_bytes: bytes = hashlib.blake2b(stage_hash_payload, digest_size=8).digest()
    stage_hash_value: int = int.from_bytes(stage_hash_bytes, byteorder="big", signed=False)

    # 32-bit space for compatibility with NumPy/Torch seed APIs.
    derived_seed: int = (
        validated_seed
        + (stage_hash_value % (2**32))
        + (validated_rank * 1_000_003)
    ) % (2**32)
    return int(derived_seed)


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader worker process deterministically.

    This function should be passed as `worker_init_fn` to DataLoader.

    Args:
        worker_id: Worker identifier provided by DataLoader.

    Raises:
        SeedConfigurationError: If worker_id is invalid.
    """
    validated_worker_id: int = _validate_non_negative_int(worker_id, key="worker_id")

    # torch.initial_seed() is already rank-aware when generator/manual seed is set
    # before DataLoader construction.
    base_seed: int = int(torch.initial_seed() % (2**32))
    worker_seed: int = (base_seed + validated_worker_id) % (2**32)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def build_torch_generator(seed: int, device: Optional[str] = None) -> torch.Generator:
    """Build a seeded torch.Generator.

    Args:
        seed: Seed value to use.
        device: Optional device identifier (e.g., `cpu`, `cuda`).

    Returns:
        Deterministically seeded torch.Generator.

    Raises:
        SeedConfigurationError: If seed is invalid.
    """
    validated_seed: int = _validate_seed_int(seed)

    generator: torch.Generator
    if device is None:
        generator = torch.Generator()
    else:
        normalized_device: str = device.strip().lower()
        try:
            generator = torch.Generator(device=normalized_device)
        except (RuntimeError, TypeError, ValueError):
            # Fallback keeps execution alive for environments where device-bound
            # generators are unsupported.
            generator = torch.Generator()

    generator.manual_seed(validated_seed)
    return generator


def validate_seed_config(cfg: Any) -> None:
    """Validate seed-related configuration values.

    Args:
        cfg: Any config-like object supporting either `.to_dict()` or mapping access.

    Raises:
        SeedConfigurationError: If seed configuration is invalid.
    """
    seed_config: SeedConfig = resolve_seed_config(cfg)
    _validate_seed_int(seed_config.global_seed)
    _normalize_stage(seed_config.stage)

    if seed_config.deterministic and seed_config.cudnn_benchmark:
        if seed_config.strict_config_validation:
            raise SeedConfigurationError(
                "Invalid deterministic setup: cudnn_benchmark=True conflicts with "
                "deterministic=True when strict_config_validation=True."
            )


def resolve_seed_config(cfg: Any, stage_override: Optional[str] = None) -> SeedConfig:
    """Resolve seed config from project config object.

    This helper reads all known seed-relevant paths with explicit defaults and
    returns a normalized immutable SeedConfig.

    Args:
        cfg: Config object (dict-like, OmegaConf, or object with `to_dict`).
        stage_override: Optional explicit stage to override config runtime.stage.

    Returns:
        Resolved SeedConfig.

    Raises:
        SeedConfigurationError: If required values are malformed.
    """
    cfg_dict: Dict[str, Any] = _to_plain_dict(cfg)

    resolved_stage: str = _normalize_stage(
        stage_override
        if stage_override is not None
        else _as_str(
            _first_present(
                cfg_dict,
                paths=(
                    ("runtime", "stage"),
                    ("runtime", "mode"),
                ),
                default=DEFAULT_STAGE,
            ),
            key_path="runtime.stage",
            default=DEFAULT_STAGE,
        )
    )

    resolved_seed: int = _as_int(
        _first_present(
            cfg_dict,
            paths=(
                ("runtime", "seed"),
                ("train_pretrain", "pretrain", "training", "seed"),
                ("pretraining", "training", "seed"),
                ("train_finetune", "finetune", "reproducibility", "seed"),
                ("downstream_public", "downstream_public", "split_policy", "seed"),
            ),
            default=DEFAULT_GLOBAL_SEED,
        ),
        key_path="seed",
        default=DEFAULT_GLOBAL_SEED,
    )

    resolved_deterministic: bool = _as_bool(
        _first_present(
            cfg_dict,
            paths=(
                ("runtime", "deterministic"),
                ("train_pretrain", "pretrain", "training", "deterministic"),
                ("train_finetune", "finetune", "reproducibility", "deterministic"),
            ),
            default=DEFAULT_DETERMINISTIC,
        ),
        key_path="deterministic",
        default=DEFAULT_DETERMINISTIC,
    )

    strict_validation: bool = _as_bool(
        _first_present(
            cfg_dict,
            paths=(("runtime", "strict_config_validation"),),
            default=True,
        ),
        key_path="runtime.strict_config_validation",
        default=True,
    )

    cudnn_benchmark: bool = _as_bool(
        _first_present(
            cfg_dict,
            paths=(("runtime", "cudnn_benchmark"),),
            default=False,
        ),
        key_path="runtime.cudnn_benchmark",
        default=False,
    )

    cudnn_deterministic: bool = _as_bool(
        _first_present(
            cfg_dict,
            paths=(("runtime", "cudnn_deterministic"),),
            default=resolved_deterministic,
        ),
        key_path="runtime.cudnn_deterministic",
        default=resolved_deterministic,
    )

    allow_tf32: bool = _as_bool(
        _first_present(
            cfg_dict,
            paths=(("runtime", "allow_tf32"),),
            default=False,
        ),
        key_path="runtime.allow_tf32",
        default=False,
    )

    return SeedConfig(
        global_seed=_validate_seed_int(resolved_seed),
        stage=resolved_stage,
        deterministic=resolved_deterministic,
        strict_config_validation=strict_validation,
        cudnn_benchmark=cudnn_benchmark,
        cudnn_deterministic=cudnn_deterministic,
        allow_tf32=allow_tf32,
    )


def configure_seeding_from_config(
    cfg: Any,
    stage_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve config and apply all seeding settings.

    Args:
        cfg: Config-like object.
        stage_override: Optional stage override.

    Returns:
        Seeding metadata dictionary for logging.

    Raises:
        SeedConfigurationError: If config is invalid.
    """
    seed_config: SeedConfig = resolve_seed_config(cfg=cfg, stage_override=stage_override)
    validate_seed_config(seed_config)
    seed_state: SeedState = _apply_seed_config(seed_config)
    return seed_state.to_dict()


def get_last_seed_state() -> Optional[Dict[str, Any]]:
    """Return last applied seed metadata, if available."""
    if _LAST_SEED_STATE is None:
        return None
    return _LAST_SEED_STATE.to_dict()


def _apply_seed_config(seed_config: SeedConfig) -> SeedState:
    """Apply seed settings to Python, NumPy, and Torch backends."""
    _validate_seed_int(seed_config.global_seed)
    _normalize_stage(seed_config.stage)

    if seed_config.deterministic and seed_config.cudnn_benchmark:
        if seed_config.strict_config_validation:
            raise SeedConfigurationError(
                "cudnn_benchmark=True conflicts with deterministic=True in strict mode."
            )
        cudnn_benchmark_effective: bool = False
    else:
        cudnn_benchmark_effective = seed_config.cudnn_benchmark

    if seed_config.deterministic and not seed_config.cudnn_deterministic:
        if seed_config.strict_config_validation:
            raise SeedConfigurationError(
                "cudnn_deterministic must be True when deterministic mode is enabled "
                "under strict validation."
            )
        cudnn_deterministic_effective: bool = True
    else:
        cudnn_deterministic_effective = seed_config.cudnn_deterministic

    rank: int = _detect_global_rank()
    world_size: int = _detect_world_size()
    stage_seed: int = derive_rank_seed(
        global_seed=seed_config.global_seed,
        stage=seed_config.stage,
        global_rank=rank,
    )

    # Set hash seed before most hash-based operations. Note that setting this
    # env var at runtime does not retroactively re-hash existing objects.
    os.environ["PYTHONHASHSEED"] = str(seed_config.global_seed)

    # Enforce deterministic cuBLAS workspace where possible.
    if seed_config.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(stage_seed)
    np.random.seed(stage_seed)

    torch.manual_seed(stage_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(stage_seed)
        torch.cuda.manual_seed_all(stage_seed)

    # Deterministic algorithm controls.
    if seed_config.deterministic:
        torch.use_deterministic_algorithms(True)
    else:
        torch.use_deterministic_algorithms(False)

    # CuDNN controls.
    torch.backends.cudnn.deterministic = bool(cudnn_deterministic_effective)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark_effective)

    # TF32 controls (CUDA only; still safe to set flags on CPU runtimes).
    torch.backends.cuda.matmul.allow_tf32 = bool(seed_config.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(seed_config.allow_tf32)

    seed_state: SeedState = SeedState(
        root_seed=seed_config.global_seed,
        stage=seed_config.stage,
        stage_seed=stage_seed,
        global_rank=rank,
        world_size=world_size,
        deterministic_mode=seed_config.deterministic,
        strict_config_validation=seed_config.strict_config_validation,
        cudnn_deterministic=bool(cudnn_deterministic_effective),
        cudnn_benchmark=bool(cudnn_benchmark_effective),
        allow_tf32=seed_config.allow_tf32,
        worker_seed_scheme="torch_initial_seed_mod_2**32_plus_worker_id",
    )

    global _LAST_SEED_STATE
    _LAST_SEED_STATE = seed_state
    return seed_state


def _detect_global_rank() -> int:
    """Detect distributed global rank from env or torch.distributed."""
    env_rank: Optional[str] = os.getenv("RANK")
    if env_rank is not None:
        try:
            return _validate_non_negative_int(int(env_rank), key="RANK")
        except (ValueError, SeedConfigurationError):
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank: int = int(torch.distributed.get_rank())
        return _validate_non_negative_int(rank, key="torch.distributed.rank")

    return 0


def _detect_world_size() -> int:
    """Detect distributed world size from env or torch.distributed."""
    env_world_size: Optional[str] = os.getenv("WORLD_SIZE")
    if env_world_size is not None:
        try:
            parsed: int = int(env_world_size)
            if parsed <= 0:
                raise SeedConfigurationError(
                    f"WORLD_SIZE must be positive, got {parsed}."
                )
            return parsed
        except (ValueError, SeedConfigurationError):
            pass

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size: int = int(torch.distributed.get_world_size())
        if world_size <= 0:
            raise SeedConfigurationError(
                f"torch.distributed.get_world_size() returned invalid value: {world_size}."
            )
        return world_size

    return 1


def _normalize_stage(stage: str) -> str:
    """Normalize and validate stage token."""
    normalized: str = stage.strip().lower()
    if normalized == "":
        raise SeedConfigurationError("Stage must be a non-empty string.")
    if normalized not in _ALLOWED_STAGES:
        raise SeedConfigurationError(
            f"Unsupported stage {stage!r}. Allowed stages: {_ALLOWED_STAGES}."
        )
    return normalized


def _validate_seed_int(seed: int) -> int:
    """Validate seed in uint32-compatible range."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SeedConfigurationError(f"Seed must be an integer, got {type(seed).__name__}.")
    if seed < 0:
        raise SeedConfigurationError(f"Seed must be non-negative, got {seed}.")
    if seed >= 2**32:
        raise SeedConfigurationError(
            f"Seed must be < 2**32 for backend compatibility, got {seed}."
        )
    return seed


def _validate_non_negative_int(value: int, key: str) -> int:
    """Validate non-negative integer values."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise SeedConfigurationError(f"{key} must be an integer, got {type(value).__name__}.")
    if value < 0:
        raise SeedConfigurationError(f"{key} must be non-negative, got {value}.")
    return value


def _to_plain_dict(cfg: Any) -> Dict[str, Any]:
    """Convert config-like object to plain dictionary."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_dict") and callable(cfg.to_dict):
        converted: Any = cfg.to_dict()
        if isinstance(converted, dict):
            return dict(converted)
        if isinstance(converted, Mapping):
            return dict(converted)

    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            container: Any = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(container, dict):
                return dict(container)
            if isinstance(container, Mapping):
                return dict(container)
    except Exception:
        pass

    raise SeedConfigurationError(
        f"Unsupported config object type for seeding resolution: {type(cfg).__name__}."
    )


def _first_present(
    data: Mapping[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any,
) -> Any:
    """Return first non-None value from nested paths."""
    for path in paths:
        value: Any = _nested_get(data, path, default=None)
        if value is not None:
            return value
    return default


def _nested_get(data: Mapping[str, Any], path: Sequence[str], default: Any) -> Any:
    """Safely get nested mapping value."""
    cursor: Any = data
    for key in path:
        if not isinstance(cursor, Mapping):
            return default
        if key not in cursor:
            return default
        cursor = cursor[key]
    return cursor


def _as_int(value: Any, key_path: str, default: int) -> int:
    """Convert value to int with explicit defaults and errors."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise SeedConfigurationError(f"Expected int at {key_path}, got bool.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise SeedConfigurationError(
                f"Expected integer-like float at {key_path}, got {value}."
            )
        return int(value)
    if isinstance(value, str):
        stripped: str = value.strip()
        if stripped == "":
            return default
        try:
            return int(stripped)
        except ValueError as exc:
            raise SeedConfigurationError(
                f"Expected int at {key_path}, got string {value!r}."
            ) from exc
    raise SeedConfigurationError(f"Expected int at {key_path}, got {type(value).__name__}.")


def _as_bool(value: Any, key_path: str, default: bool) -> bool:
    """Convert value to bool with explicit defaults and errors."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise SeedConfigurationError(
        f"Expected bool at {key_path}, got {type(value).__name__}: {value!r}."
    )


def _as_str(value: Any, key_path: str, default: str) -> str:
    """Convert value to string with defaults."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    raise SeedConfigurationError(
        f"Expected string at {key_path}, got {type(value).__name__}."
    )


__all__ = [
    "SEED_CONTRACT_VERSION",
    "DEFAULT_GLOBAL_SEED",
    "DEFAULT_STAGE",
    "DEFAULT_DETERMINISTIC",
    "SeedingError",
    "SeedConfigurationError",
    "SeedConfig",
    "SeedState",
    "seed_everything",
    "derive_rank_seed",
    "seed_worker",
    "build_torch_generator",
    "validate_seed_config",
    "resolve_seed_config",
    "configure_seeding_from_config",
    "get_last_seed_state",
]
