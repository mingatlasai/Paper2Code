"""Stage-3 orchestration script for TITAN multimodal WSI-report pretraining.

This script is intentionally thin and only handles:
- configuration loading/overrides,
- deterministic/distributed runtime setup,
- registry-driven dataset/trainer construction,
- required stage-2 initialization checkpoint loading,
- optional stage-3 checkpoint resume,
- stage-3 training dispatch.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.core.config_schema import (
    ConfigLoadError,
    ConfigValidationError,
    ExperimentConfig,
)
from src.core.registry import Registry
from src.core.utils import (
    cleanup_distributed,
    init_distributed,
    safe_symlink_or_copy,
    seed_everything,
    seed_worker,
    validate_repro_constants,
)
from src.data.collate import build_collate_fn


def _seed_worker_entry(worker_id: int) -> None:
    """Top-level worker seed entry to keep DataLoader picklable."""
    raw_seed: str = os.environ.get("TITAN_SEED", str(_DEFAULT_SEED))
    seed_value: int = int(raw_seed)
    seed_worker(worker_id=worker_id, base_seed=seed_value)


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml/task contract.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: tuple[int, int] = (64, 64)

_STAGE3_NUM_PAIRS: int = 182_862
_STAGE3_BATCH_SIZE_PER_GPU: int = 16
_STAGE3_GRAD_ACCUM: int = 2
_STAGE3_EFFECTIVE_BATCH_SIZE: int = 256

_DEFAULT_CONFIG_PATH: str = "configs/default.yaml"
_FALLBACK_STAGE3_CONFIG_PATH: str = "configs/train/stage3_coca.yaml"

_DEFAULT_NUM_WORKERS: int = 8
_DEFAULT_PIN_MEMORY: bool = True
_DEFAULT_SEED: int = 42


class RunStage3Error(RuntimeError):
    """Base exception for stage-3 orchestration failures."""


@dataclass(frozen=True)
class Stage3CliArgs:
    """CLI arguments for stage-3 execution."""

    config: str
    init: str
    resume: Optional[str]
    output_dir: Optional[str]
    seed: Optional[int]
    batch_size: Optional[int]
    grad_accum_steps: Optional[int]
    num_workers: Optional[int]
    mixed_precision: Optional[bool]
    disable_mixed_precision: bool


def _parse_args() -> Stage3CliArgs:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Run TITAN stage-3 (WSI-report CoCa alignment).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to experiment config YAML.",
    )
    parser.add_argument(
        "--init",
        type=str,
        required=True,
        help="Required stage-2 checkpoint path (titan_stage2.ckpt lineage).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional stage-3 checkpoint path to resume training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output root directory for this run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override deterministic seed.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override local batch size per GPU.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=None,
        help="Override gradient accumulation steps.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override dataloader worker count.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Force-enable mixed precision.",
    )
    parser.add_argument(
        "--disable-mixed-precision",
        action="store_true",
        help="Force-disable mixed precision.",
    )

    namespace: argparse.Namespace = parser.parse_args()
    return Stage3CliArgs(
        config=str(namespace.config),
        init=str(namespace.init),
        resume=(str(namespace.resume) if namespace.resume else None),
        output_dir=(str(namespace.output_dir) if namespace.output_dir else None),
        seed=(int(namespace.seed) if namespace.seed is not None else None),
        batch_size=(int(namespace.batch_size) if namespace.batch_size is not None else None),
        grad_accum_steps=(
            int(namespace.grad_accum_steps) if namespace.grad_accum_steps is not None else None
        ),
        num_workers=(int(namespace.num_workers) if namespace.num_workers is not None else None),
        mixed_precision=(True if bool(namespace.mixed_precision) else None),
        disable_mixed_precision=bool(namespace.disable_mixed_precision),
    )


def _resolve_config_path(requested_path: str) -> Path:
    candidate: Path = Path(requested_path).expanduser().resolve()
    if candidate.exists() and candidate.is_file():
        return candidate

    requested_name: str = str(requested_path).strip()
    if requested_name == _DEFAULT_CONFIG_PATH:
        fallback: Path = Path(_FALLBACK_STAGE3_CONFIG_PATH).expanduser().resolve()
        if fallback.exists() and fallback.is_file():
            return fallback

    raise FileNotFoundError(
        "Config file not found. "
        f"requested='{requested_path}', fallback='{_FALLBACK_STAGE3_CONFIG_PATH}'."
    )


def _load_and_override_config(args: Stage3CliArgs) -> ExperimentConfig:
    resolved_cfg_path: Path = _resolve_config_path(args.config)

    try:
        cfg: ExperimentConfig = ExperimentConfig.from_yaml(str(resolved_cfg_path))
    except (ConfigLoadError, ConfigValidationError) as exc:
        raise RunStage3Error(f"Failed loading config from '{resolved_cfg_path}': {exc}") from exc

    # Enforce stage/mode for this runner.
    cfg.mode = "train_stage3"
    cfg.stage = "stage3_wsi_report_alignment"

    # CLI overrides.
    if args.seed is not None:
        cfg.runtime.seed = int(args.seed)
    if args.num_workers is not None:
        cfg.runtime.num_workers = int(args.num_workers)

    if args.output_dir is not None:
        cfg.paths.output_root = str(Path(args.output_dir).expanduser().resolve())

    stage_cfg = cfg.training.stage3_wsi_report_alignment
    if args.batch_size is not None:
        stage_cfg.batch_size = int(args.batch_size)
    if args.grad_accum_steps is not None:
        stage_cfg.grad_accum_steps = int(args.grad_accum_steps)

    if args.disable_mixed_precision:
        stage_cfg.mixed_precision = False
    elif args.mixed_precision is not None:
        stage_cfg.mixed_precision = bool(args.mixed_precision)

    # Recompute effective batch from configured stage3 hardware.
    stage_cfg.effective_batch_size = (
        int(stage_cfg.batch_size)
        * int(cfg.hardware.stage3.gpus)
        * int(stage_cfg.grad_accum_steps)
    )

    # Validate full config and paper constants after overrides.
    try:
        cfg.validate()
    except ConfigValidationError as exc:
        raise RunStage3Error(f"Configuration validation failed: {exc}") from exc

    validate_repro_constants(cfg)
    _validate_stage3_invariants(cfg)

    return cfg


def _validate_stage3_invariants(cfg: ExperimentConfig) -> None:
    data_cfg = cfg.data
    model_cfg = cfg.model.slide_encoder
    multimodal_cfg = cfg.model.multimodal
    stage_cfg = cfg.training.stage3_wsi_report_alignment

    if int(data_cfg.patch_size) != _PATCH_SIZE_PX:
        raise RunStage3Error(f"patch_size must be {_PATCH_SIZE_PX}, got {data_cfg.patch_size}.")
    if str(data_cfg.magnification) != _MAGNIFICATION:
        raise RunStage3Error(
            f"magnification must be '{_MAGNIFICATION}', got '{data_cfg.magnification}'."
        )
    if int(data_cfg.feature_dim) != _FEATURE_DIM:
        raise RunStage3Error(f"feature_dim must be {_FEATURE_DIM}, got {data_cfg.feature_dim}.")

    if tuple(data_cfg.roi_region_grid_size) != _STAGE1_REGION_GRID:
        raise RunStage3Error(
            "roi_region_grid_size mismatch: "
            f"expected {_STAGE1_REGION_GRID}, got {tuple(data_cfg.roi_region_grid_size)}"
        )
    if tuple(data_cfg.stage3_wsi_crop_grid_size) != _STAGE3_CROP_GRID:
        raise RunStage3Error(
            "stage3_wsi_crop_grid_size mismatch: "
            f"expected {_STAGE3_CROP_GRID}, got {tuple(data_cfg.stage3_wsi_crop_grid_size)}"
        )

    if int(model_cfg.embedding_dim) != _FEATURE_DIM:
        raise RunStage3Error(
            "model embedding_dim must match feature_dim=768, "
            f"got {model_cfg.embedding_dim}."
        )
    if int(model_cfg.num_attention_heads) * int(model_cfg.head_dim) != int(model_cfg.embedding_dim):
        raise RunStage3Error("num_attention_heads * head_dim must equal embedding_dim.")

    if int(multimodal_cfg.reconstruction_queries) != 128:
        raise RunStage3Error(
            "reconstruction_queries must be 128, "
            f"got {multimodal_cfg.reconstruction_queries}."
        )
    if int(multimodal_cfg.text_embedding_dim) != int(model_cfg.embedding_dim):
        raise RunStage3Error(
            "multimodal text embedding dim must match vision embedding dim. "
            f"text={multimodal_cfg.text_embedding_dim}, vision={model_cfg.embedding_dim}."
        )
    if int(multimodal_cfg.text_encoder_layers) != 12 or int(multimodal_cfg.text_decoder_layers) != 12:
        raise RunStage3Error("text encoder/decoder layers must both be 12 for stage3.")

    if int(stage_cfg.batch_size) != _STAGE3_BATCH_SIZE_PER_GPU:
        raise RunStage3Error(
            f"stage3 batch_size must be {_STAGE3_BATCH_SIZE_PER_GPU}, got {stage_cfg.batch_size}."
        )
    if int(stage_cfg.grad_accum_steps) != _STAGE3_GRAD_ACCUM:
        raise RunStage3Error(
            f"stage3 grad_accum_steps must be {_STAGE3_GRAD_ACCUM}, got {stage_cfg.grad_accum_steps}."
        )
    if int(stage_cfg.effective_batch_size) != _STAGE3_EFFECTIVE_BATCH_SIZE:
        raise RunStage3Error(
            "stage3 effective_batch_size must be "
            f"{_STAGE3_EFFECTIVE_BATCH_SIZE}, got {stage_cfg.effective_batch_size}."
        )

    if int(stage_cfg.num_pairs) != _STAGE3_NUM_PAIRS:
        raise RunStage3Error(
            f"stage3 num_pairs must be {_STAGE3_NUM_PAIRS}, got {stage_cfg.num_pairs}."
        )


def _validate_stage2_init_checkpoint(path: Path, cfg: ExperimentConfig) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Stage-2 init checkpoint not found: {path}")

    try:
        payload: Any = torch.load(path, map_location="cpu")
    except Exception as exc:  # noqa: BLE001
        raise RunStage3Error(f"Failed reading init checkpoint '{path}': {exc}") from exc

    if not isinstance(payload, Mapping):
        raise RunStage3Error(
            "Init checkpoint payload must be a mapping with model weights. "
            f"Got type={type(payload).__name__}."
        )

    has_state_dict: bool = (
        "model_state_dict" in payload
        or "state_dict" in payload
        or any(isinstance(key, str) and key.endswith("weight") for key in payload.keys())
    )
    if not has_state_dict:
        raise RunStage3Error(
            "Init checkpoint does not contain recognizable state_dict fields."
        )

    # Optional lineage check when trainer extra state is present.
    extra_state_obj: Any = payload.get("extra_state")
    if isinstance(extra_state_obj, Mapping):
        stage_name_obj: Any = extra_state_obj.get("stage")
        if stage_name_obj is not None:
            stage_name: str = str(stage_name_obj)
            allowed_stages: set[str] = {
                "stage2_roi_caption_alignment",
                "stage3_wsi_report_alignment",
            }
            if stage_name not in allowed_stages:
                raise RunStage3Error(
                    "Init checkpoint stage mismatch. "
                    f"Expected one of {sorted(allowed_stages)}, got '{stage_name}'."
                )

    # Optional strict architecture check when model_config is present.
    model_config_obj: Any = payload.get("model_config")
    if isinstance(model_config_obj, Mapping):
        embed_dim_value: int = int(model_config_obj.get("embed_dim", model_config_obj.get("embedding_dim", 768)))
        num_layers_value: int = int(model_config_obj.get("num_layers", 6))
        num_heads_value: int = int(model_config_obj.get("num_heads", model_config_obj.get("num_attention_heads", 12)))
        head_dim_value: int = int(model_config_obj.get("head_dim", 64))
        mlp_dim_value: int = int(model_config_obj.get("mlp_dim", model_config_obj.get("mlp_hidden_dim", 3072)))

        if embed_dim_value != int(cfg.model.slide_encoder.embedding_dim):
            raise RunStage3Error(
                "Init checkpoint embed dim mismatch. "
                f"checkpoint={embed_dim_value}, config={cfg.model.slide_encoder.embedding_dim}."
            )
        if num_layers_value != int(cfg.model.slide_encoder.num_layers):
            raise RunStage3Error(
                "Init checkpoint num_layers mismatch. "
                f"checkpoint={num_layers_value}, config={cfg.model.slide_encoder.num_layers}."
            )
        if num_heads_value != int(cfg.model.slide_encoder.num_attention_heads):
            raise RunStage3Error(
                "Init checkpoint num_attention_heads mismatch. "
                f"checkpoint={num_heads_value}, config={cfg.model.slide_encoder.num_attention_heads}."
            )
        if head_dim_value != int(cfg.model.slide_encoder.head_dim):
            raise RunStage3Error(
                "Init checkpoint head_dim mismatch. "
                f"checkpoint={head_dim_value}, config={cfg.model.slide_encoder.head_dim}."
            )
        if mlp_dim_value != int(cfg.model.slide_encoder.mlp_hidden_dim):
            raise RunStage3Error(
                "Init checkpoint mlp_hidden_dim mismatch. "
                f"checkpoint={mlp_dim_value}, config={cfg.model.slide_encoder.mlp_hidden_dim}."
            )


def _extract_state_dict_for_init(checkpoint_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    state_dict_obj: Any = checkpoint_payload.get("model_state_dict")
    if isinstance(state_dict_obj, Mapping):
        return state_dict_obj

    state_dict_obj = checkpoint_payload.get("state_dict")
    if isinstance(state_dict_obj, Mapping):
        return state_dict_obj

    # Raw state dict fallback.
    if all(isinstance(key, str) for key in checkpoint_payload.keys()):
        return checkpoint_payload

    raise RunStage3Error("Checkpoint does not contain a valid state dict mapping.")


def _normalize_state_dict_keys(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        if not isinstance(raw_key, str):
            continue
        key: str = raw_key
        if key.startswith("module."):
            key = key[len("module.") :]
        if key.startswith("model."):
            key = key[len("model.") :]
        normalized[key] = value
    return normalized


def _build_train_loader(cfg: ExperimentConfig, registry: Registry, rank: int, world_size: int) -> DataLoader:
    dataset = registry.create_dataset("stage3_dataset", cfg.data)
    collate_fn = build_collate_fn(mode="train_stage3")

    distributed: bool = bool(world_size > 1)
    sampler: Optional[DistributedSampler] = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=False,
        )

    num_workers: int = int(cfg.runtime.num_workers)
    if num_workers < 0:
        num_workers = _DEFAULT_NUM_WORKERS

    pin_memory: bool = bool(getattr(cfg.runtime, "pin_memory", _DEFAULT_PIN_MEMORY))

    generator: torch.Generator = torch.Generator()
    generator.manual_seed(int(cfg.runtime.seed))

    loader: DataLoader = DataLoader(
        dataset=dataset,
        batch_size=int(cfg.training.stage3_wsi_report_alignment.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=bool(num_workers > 0),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker_entry,
        generator=generator,
    )
    return loader


def _resolve_resume_path(resume: Optional[str]) -> Optional[Path]:
    if resume is None:
        return None
    checkpoint_path: Path = Path(resume).expanduser().resolve()
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _initialize_stage3_from_stage2(trainer: Any, init_path: Path) -> None:
    model_obj: Any = getattr(trainer, "model", None)
    if model_obj is None:
        raise RunStage3Error("Trainer does not expose a 'model' attribute.")

    checkpoint_payload: Any = torch.load(init_path, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise RunStage3Error("Stage-2 checkpoint payload must be a mapping.")

    state_dict_raw: Mapping[str, Any] = _extract_state_dict_for_init(checkpoint_payload)
    state_dict: Dict[str, Any] = _normalize_state_dict_keys(state_dict_raw)

    # Strict load to guarantee architecture compatibility for stage handoff.
    missing_keys, unexpected_keys = model_obj.load_state_dict(state_dict, strict=False)

    loaded_keys: int = len(state_dict)
    if loaded_keys <= 0:
        raise RunStage3Error("No parameters loaded from stage-2 checkpoint.")

    # Expect very high overlap with same architecture; fail if too sparse.
    total_keys: int = len(model_obj.state_dict())
    if loaded_keys < max(1, int(0.10 * total_keys)):
        raise RunStage3Error(
            "Loaded too few parameters from stage-2 checkpoint. "
            f"loaded={loaded_keys}, model_keys={total_keys}, "
            f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}."
        )


def run_stage3(args: Stage3CliArgs) -> int:
    cfg: ExperimentConfig = _load_and_override_config(args)

    # BaseTrainer reads these through environment fallbacks for TrainConfig extras.
    os.environ["TITAN_OUTPUT_ROOT"] = str(Path(cfg.paths.output_root).expanduser().resolve())
    os.environ["TITAN_SEED"] = str(int(cfg.runtime.seed))

    seed_everything(seed=int(cfg.runtime.seed), deterministic=bool(cfg.runtime.deterministic))

    distributed_enabled: bool = bool(cfg.runtime.distributed.enabled)
    ddp_context = None

    init_path: Path = Path(args.init).expanduser().resolve()
    _validate_stage2_init_checkpoint(path=init_path, cfg=cfg)

    try:
        if distributed_enabled:
            ddp_context = init_distributed(
                backend=str(cfg.runtime.distributed.backend),
                init_method=str(cfg.runtime.distributed.init_method),
            )
            rank: int = int(ddp_context.rank)
            world_size: int = int(ddp_context.world_size)
        else:
            rank = 0
            world_size = 1

        registry: Registry = Registry(auto_register_defaults=True)

        train_loader: DataLoader = _build_train_loader(
            cfg=cfg,
            registry=registry,
            rank=rank,
            world_size=world_size,
        )

        trainer = registry.create_trainer(
            name="stage3_trainer",
            cfg=cfg.training.stage3_wsi_report_alignment,
        )

        resume_path: Optional[Path] = _resolve_resume_path(args.resume)
        if resume_path is not None:
            trainer.load_checkpoint(str(resume_path))
        else:
            _initialize_stage3_from_stage2(trainer=trainer, init_path=init_path)

        trainer.fit(train_loader=train_loader, val_loader=None)

        # Ensure canonical artifact alias under configured output root.
        final_ckpt: Path = trainer.run_dirs["checkpoints"] / cfg.artifacts.stage3_checkpoint_name
        alias_ckpt: Path = Path(cfg.paths.output_root).expanduser().resolve() / cfg.artifacts.stage3_checkpoint_name
        if final_ckpt.exists() and final_ckpt.is_file() and final_ckpt.resolve() != alias_ckpt.resolve():
            safe_symlink_or_copy(src=str(final_ckpt), dst=str(alias_ckpt))

        return 0
    finally:
        cleanup_distributed(ddp_context)


def main() -> None:
    args: Stage3CliArgs = _parse_args()
    try:
        exit_code: int = run_stage3(args)
    except Exception as exc:  # noqa: BLE001
        raise RunStage3Error(f"Stage-3 run failed: {exc}") from exc
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
