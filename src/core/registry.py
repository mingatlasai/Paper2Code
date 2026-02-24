"""Factory registry for TITAN reproduction components.

This module provides a strict, config-driven registry for datasets, models, and
trainers. It follows the design contract and keeps creation logic centralized to
avoid hardcoded branching in the CLI/application layers.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import inspect
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar

import torch

from src.core.config_schema import DataConfig, ModelConfig, TrainConfig


# -----------------------------------------------------------------------------
# Paper/config locked constants from provided config contract.
# -----------------------------------------------------------------------------
_PATCH_SIZE: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768
_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_GRID: Tuple[int, int] = (64, 64)

_EMBED_DIM: int = 768
_NUM_LAYERS: int = 6
_NUM_HEADS: int = 12
_HEAD_DIM: int = 64
_MLP_DIM: int = 3072

_STAGE1_BATCH: int = 256
_STAGE1_GPUS: int = 4
_STAGE1_ACCUM: int = 1
_STAGE1_EFFECTIVE: int = 1024

_STAGE2_BATCH: int = 196
_STAGE2_GPUS: int = 8
_STAGE2_ACCUM: int = 2
_STAGE2_EFFECTIVE: int = 3136

_STAGE3_BATCH: int = 16
_STAGE3_GPUS: int = 8
_STAGE3_ACCUM: int = 2
_STAGE3_EFFECTIVE: int = 256


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class RegistryError(RuntimeError):
    """Base exception for registry failures."""


class RegistryRegistrationError(RegistryError):
    """Raised when registration fails."""


class RegistryLookupError(RegistryError):
    """Raised when a component key cannot be resolved."""


class RegistryConstructionError(RegistryError):
    """Raised when object construction fails."""


class RegistryInvariantError(RegistryConstructionError):
    """Raised when required paper/config invariants are violated."""


T = TypeVar("T")


@dataclass(frozen=True)
class _Target:
    """Lazy import descriptor for a registered class."""

    module_path: str
    class_name: str


class Registry:
    """Central registry for datasets, models, and trainers.

    Public interface follows the design contract:
    - register_dataset(name: str, cls: type) -> None
    - register_model(name: str, cls: type) -> None
    - register_trainer(name: str, cls: type) -> None
    - create_dataset(name: str, cfg: DataConfig) -> BaseDataset
    - create_model(name: str, cfg: ModelConfig) -> torch.nn.Module
    - create_trainer(name: str, cfg: TrainConfig) -> BaseTrainer
    """

    def __init__(self, auto_register_defaults: bool = True) -> None:
        self._dataset_classes: Dict[str, Type[Any]] = {}
        self._model_classes: Dict[str, Type[Any]] = {}
        self._trainer_classes: Dict[str, Type[Any]] = {}

        self._dataset_targets: Dict[str, _Target] = {}
        self._model_targets: Dict[str, _Target] = {}
        self._trainer_targets: Dict[str, _Target] = {}

        if auto_register_defaults:
            self._register_default_targets()

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------
    def register_dataset(self, name: str, cls: type) -> None:
        """Register a dataset class under a stable key."""
        key: str = self._normalize_key(name)
        self._validate_class_registration(kind="dataset", key=key, cls=cls)
        self._dataset_classes[key] = cls

    def register_model(self, name: str, cls: type) -> None:
        """Register a model class under a stable key."""
        key: str = self._normalize_key(name)
        self._validate_class_registration(kind="model", key=key, cls=cls)
        self._model_classes[key] = cls

    def register_trainer(self, name: str, cls: type) -> None:
        """Register a trainer class under a stable key."""
        key: str = self._normalize_key(name)
        self._validate_class_registration(kind="trainer", key=key, cls=cls)
        self._trainer_classes[key] = cls

    # ------------------------------------------------------------------
    # Public factory API
    # ------------------------------------------------------------------
    def create_dataset(self, name: str, cfg: DataConfig) -> Any:
        """Create a dataset instance from its key and DataConfig."""
        self._validate_data_config(cfg)
        key: str = self._normalize_key(name)
        dataset_cls: Type[Any] = self._resolve_dataset_class(key)

        if key == "stage1_dataset":
            kwargs: Dict[str, Any] = {
                "meta_csv": cfg.meta_csv,
                "groups_path": self._resolve_stage1_groups_path(cfg),
                "crop_size": int(cfg.roi_region_grid_size[0]),
            }
        elif key == "stage2_dataset":
            kwargs = {
                "pairs_csv": cfg.manifests.roi_caption_pairs_jsonl,
                "tokenizer_name": self._default_text_encoder_source(),
            }
        elif key == "stage3_dataset":
            kwargs = {
                "pairs_csv": cfg.manifests.wsi_report_pairs_jsonl,
                "tokenizer_name": self._default_text_encoder_source(),
                "crop_hw": tuple(cfg.stage3_wsi_crop_grid_size),
            }
        else:
            kwargs = {"cfg": cfg}

        try:
            return self._instantiate_with_signature(dataset_cls, kwargs=kwargs, context=f"dataset:{key}")
        except Exception as exc:  # pragma: no cover - defensive path
            raise RegistryConstructionError(f"Failed constructing dataset '{key}'.") from exc

    def create_model(self, name: str, cfg: ModelConfig) -> torch.nn.Module:
        """Create a model instance from its key and ModelConfig."""
        self._validate_model_config(cfg)
        key: str = self._normalize_key(name)
        model_cls: Type[Any] = self._resolve_model_class(key)

        if key == "titan_encoder":
            kwargs: Dict[str, Any] = {"cfg": cfg}
            try:
                model_obj: Any = self._instantiate_with_signature(model_cls, kwargs=kwargs, context=f"model:{key}")
            except Exception as exc:  # pragma: no cover - defensive path
                raise RegistryConstructionError(f"Failed constructing model '{key}'.") from exc
            if not isinstance(model_obj, torch.nn.Module):
                raise RegistryConstructionError(
                    f"Model '{key}' must be a torch.nn.Module, got {type(model_obj).__name__}."
                )
            return model_obj

        if key == "coca_multimodal":
            vision_model: torch.nn.Module = self.create_model("titan_encoder", cfg)
            text_encoder_obj: Any = self._build_text_encoder_fallback(cfg)
            decoder_obj: Any = self._build_multimodal_decoder_fallback(cfg)

            kwargs = {
                "vision": vision_model,
                "text_encoder": text_encoder_obj,
                "decoder": decoder_obj,
            }
            try:
                model_obj = self._instantiate_with_signature(model_cls, kwargs=kwargs, context=f"model:{key}")
            except Exception as exc:
                raise RegistryConstructionError(
                    "Failed constructing model 'coca_multimodal'. "
                    "Ensure TextEncoder/MultimodalDecoder classes exist and are compatible."
                ) from exc
            if not isinstance(model_obj, torch.nn.Module):
                raise RegistryConstructionError(
                    f"Model '{key}' must be a torch.nn.Module, got {type(model_obj).__name__}."
                )
            return model_obj

        if key == "text_encoder":
            text_encoder_obj = self._build_text_encoder_fallback(cfg)
            if not isinstance(text_encoder_obj, torch.nn.Module):
                raise RegistryConstructionError(
                    f"Model '{key}' must be a torch.nn.Module, got {type(text_encoder_obj).__name__}."
                )
            return text_encoder_obj

        raise RegistryLookupError(
            f"Unknown model key '{key}'. Registered keys: {sorted(self._all_model_keys())}"
        )

    def create_trainer(self, name: str, cfg: TrainConfig) -> Any:
        """Create a trainer instance from its key and TrainConfig."""
        key: str = self._normalize_key(name)
        trainer_cls: Type[Any] = self._resolve_trainer_class(key)
        self._validate_train_config(cfg, trainer_key=key)

        default_model_cfg: ModelConfig = ModelConfig()

        if key == "stage1_trainer":
            model_obj: torch.nn.Module = self.create_model("titan_encoder", default_model_cfg)
            ibot_head_obj: Any = self._build_ibot_head_fallback(embed_dim=default_model_cfg.embed_dim)
            ibot_loss_obj: Any = self._build_ibot_loss_fallback(cfg)
            kwargs: Dict[str, Any] = {
                "model": model_obj,
                "ibot_head": ibot_head_obj,
                "loss_fn": ibot_loss_obj,
                "cfg": cfg,
            }
            return self._instantiate_with_signature(trainer_cls, kwargs=kwargs, context=f"trainer:{key}")

        if key in {"stage2_trainer", "stage3_trainer"}:
            model_obj = self.create_model("coca_multimodal", default_model_cfg)
            kwargs = {
                "model": model_obj,
                "cfg": cfg,
            }
            return self._instantiate_with_signature(trainer_cls, kwargs=kwargs, context=f"trainer:{key}")

        return self._instantiate_with_signature(trainer_cls, kwargs={"cfg": cfg}, context=f"trainer:{key}")

    # ------------------------------------------------------------------
    # Internal registration defaults
    # ------------------------------------------------------------------
    def _register_default_targets(self) -> None:
        self._dataset_targets.update(
            {
                "stage1_dataset": _Target("src.data.datasets", "Stage1Dataset"),
                "stage2_dataset": _Target("src.data.datasets", "Stage2Dataset"),
                "stage3_dataset": _Target("src.data.datasets", "Stage3Dataset"),
            }
        )
        self._model_targets.update(
            {
                "titan_encoder": _Target("src.models.titan_encoder", "TITANEncoder"),
                "coca_multimodal": _Target("src.models.coca_multimodal", "CoCaModel"),
                "text_encoder": _Target("src.models.text_modules", "TextEncoder"),
            }
        )
        self._trainer_targets.update(
            {
                "stage1_trainer": _Target("src.train.stage1_trainer", "Stage1Trainer"),
                "stage2_trainer": _Target("src.train.stage2_trainer", "Stage2Trainer"),
                "stage3_trainer": _Target("src.train.stage3_trainer", "Stage3Trainer"),
            }
        )

    # ------------------------------------------------------------------
    # Internal class resolution
    # ------------------------------------------------------------------
    def _resolve_dataset_class(self, key: str) -> Type[Any]:
        if key in self._dataset_classes:
            return self._dataset_classes[key]
        target: Optional[_Target] = self._dataset_targets.get(key)
        if target is None:
            raise RegistryLookupError(
                f"Unknown dataset key '{key}'. Registered keys: {sorted(self._all_dataset_keys())}"
            )
        cls: Type[Any] = self._import_target(target=target)
        self._dataset_classes[key] = cls
        return cls

    def _resolve_model_class(self, key: str) -> Type[Any]:
        if key in self._model_classes:
            return self._model_classes[key]
        target = self._model_targets.get(key)
        if target is None:
            raise RegistryLookupError(
                f"Unknown model key '{key}'. Registered keys: {sorted(self._all_model_keys())}"
            )
        cls = self._import_target(target=target)
        self._model_classes[key] = cls
        return cls

    def _resolve_trainer_class(self, key: str) -> Type[Any]:
        if key in self._trainer_classes:
            return self._trainer_classes[key]
        target = self._trainer_targets.get(key)
        if target is None:
            raise RegistryLookupError(
                f"Unknown trainer key '{key}'. Registered keys: {sorted(self._all_trainer_keys())}"
            )
        cls = self._import_target(target=target)
        self._trainer_classes[key] = cls
        return cls

    def _import_target(self, target: _Target) -> Type[Any]:
        try:
            module = importlib.import_module(target.module_path)
        except Exception as exc:  # pragma: no cover - depends on runtime/module availability
            raise RegistryLookupError(
                f"Failed importing module '{target.module_path}' for '{target.class_name}'."
            ) from exc

        cls: Any = getattr(module, target.class_name, None)
        if cls is None:
            raise RegistryLookupError(
                f"Module '{target.module_path}' does not define class '{target.class_name}'."
            )
        if not inspect.isclass(cls):
            raise RegistryLookupError(
                f"Resolved target '{target.module_path}.{target.class_name}' is not a class."
            )
        return cls

    # ------------------------------------------------------------------
    # Internal builders for stage-specific dependencies
    # ------------------------------------------------------------------
    def _build_ibot_head_fallback(self, embed_dim: int) -> Any:
        target = _Target("src.models.ibot_heads", "IBOTHead")
        ibot_head_cls: Type[Any] = self._import_target(target)
        kwargs: Dict[str, Any] = {
            "embed_dim": int(embed_dim),
            "out_dim": int(embed_dim),
        }
        return self._instantiate_with_signature(ibot_head_cls, kwargs=kwargs, context="model:ibot_head")

    def _build_ibot_loss_fallback(self, train_cfg: TrainConfig) -> Any:
        target = _Target("src.models.losses", "IBOTLoss")
        ibot_loss_cls: Type[Any] = self._import_target(target)

        student_temp: Optional[float] = self._safe_nested_get(train_cfg, ("ibot", "student_temperature"))
        teacher_temp: Optional[float] = self._safe_nested_get(train_cfg, ("ibot", "teacher_temperature"))
        center_momentum: Optional[float] = self._safe_nested_get(train_cfg, ("ibot", "center_momentum"))

        kwargs: Dict[str, Any] = {
            "student_temp": student_temp,
            "teacher_temp": teacher_temp,
            "center_momentum": center_momentum,
        }
        return self._instantiate_with_signature(ibot_loss_cls, kwargs=kwargs, context="model:ibot_loss")

    def _build_text_encoder_fallback(self, cfg: ModelConfig) -> Any:
        text_encoder_cls: Type[Any] = self._resolve_model_class("text_encoder")

        # These are intentionally unresolved in the supplied paper/config. We pass
        # conservative constructor defaults only when the class requires them.
        kwargs: Dict[str, Any] = {
            "name": self._default_text_encoder_source(),
            "vocab_size": 1,
            "max_len": 1,
        }

        if hasattr(cfg, "embed_dim"):
            kwargs["embed_dim"] = int(cfg.embed_dim)

        try:
            return self._instantiate_with_signature(text_encoder_cls, kwargs=kwargs, context="model:text_encoder")
        except Exception as exc:
            raise RegistryConstructionError(
                "Failed constructing TextEncoder. Provide a compatible implementation "
                "or register a custom text_encoder class explicitly."
            ) from exc

    def _build_multimodal_decoder_fallback(self, cfg: ModelConfig) -> Any:
        target = _Target("src.models.text_modules", "MultimodalDecoder")
        decoder_cls: Type[Any] = self._import_target(target)
        kwargs: Dict[str, Any] = {
            "num_layers": 12,
            "embed_dim": int(cfg.embed_dim),
        }
        return self._instantiate_with_signature(decoder_cls, kwargs=kwargs, context="model:multimodal_decoder")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_class_registration(self, kind: str, key: str, cls: type) -> None:
        if not key:
            raise RegistryRegistrationError(f"{kind} key must be a non-empty string.")
        if not inspect.isclass(cls):
            raise RegistryRegistrationError(f"{kind} registration for key '{key}' must be a class.")

        store: MutableMapping[str, Type[Any]]
        if kind == "dataset":
            store = self._dataset_classes
        elif kind == "model":
            store = self._model_classes
        elif kind == "trainer":
            store = self._trainer_classes
        else:  # pragma: no cover - internal misuse guard
            raise RegistryRegistrationError(f"Unsupported registration kind: {kind!r}")

        existing = store.get(key)
        if existing is not None and existing is not cls:
            raise RegistryRegistrationError(
                f"Duplicate registration for {kind} key '{key}'. "
                f"Existing class: {existing.__name__}, new class: {cls.__name__}."
            )

    def _validate_data_config(self, cfg: DataConfig) -> None:
        if int(cfg.patch_size) != _PATCH_SIZE:
            raise RegistryInvariantError(f"data.patch_size must be {_PATCH_SIZE}, got {cfg.patch_size!r}.")
        if str(cfg.magnification) != _MAGNIFICATION:
            raise RegistryInvariantError(
                f"data.magnification must be '{_MAGNIFICATION}', got {cfg.magnification!r}."
            )
        if int(cfg.feature_dim) != _FEATURE_DIM:
            raise RegistryInvariantError(f"data.feature_dim must be {_FEATURE_DIM}, got {cfg.feature_dim!r}.")

        if tuple(cfg.roi_region_grid_size) != _STAGE1_REGION_GRID:
            raise RegistryInvariantError(
                f"data.roi_region_grid_size must be {_STAGE1_REGION_GRID}, got {cfg.roi_region_grid_size!r}."
            )
        if tuple(cfg.stage3_wsi_crop_grid_size) != _STAGE3_GRID:
            raise RegistryInvariantError(
                f"data.stage3_wsi_crop_grid_size must be {_STAGE3_GRID}, got {cfg.stage3_wsi_crop_grid_size!r}."
            )

    def _validate_model_config(self, cfg: ModelConfig) -> None:
        if int(cfg.embed_dim) != _EMBED_DIM:
            raise RegistryInvariantError(f"model.embed_dim must be {_EMBED_DIM}, got {cfg.embed_dim!r}.")
        if int(cfg.num_layers) != _NUM_LAYERS:
            raise RegistryInvariantError(f"model.num_layers must be {_NUM_LAYERS}, got {cfg.num_layers!r}.")
        if int(cfg.num_heads) != _NUM_HEADS:
            raise RegistryInvariantError(f"model.num_heads must be {_NUM_HEADS}, got {cfg.num_heads!r}.")
        if int(cfg.mlp_dim) != _MLP_DIM:
            raise RegistryInvariantError(f"model.mlp_dim must be {_MLP_DIM}, got {cfg.mlp_dim!r}.")

        head_dim: int = int(getattr(cfg, "head_dim", _HEAD_DIM))
        if head_dim != _HEAD_DIM:
            raise RegistryInvariantError(f"model.head_dim must be {_HEAD_DIM}, got {head_dim!r}.")
        if int(cfg.num_heads) * head_dim != int(cfg.embed_dim):
            raise RegistryInvariantError(
                "model invariant failed: num_heads * head_dim must equal embed_dim."
            )

    def _validate_train_config(self, cfg: TrainConfig, trainer_key: str) -> None:
        stage: str = str(cfg.stage)

        if trainer_key == "stage1_trainer" and stage != "stage1_titan_v":
            raise RegistryInvariantError(
                f"trainer '{trainer_key}' requires cfg.stage='stage1_titan_v', got {stage!r}."
            )
        if trainer_key == "stage2_trainer" and stage != "stage2_roi_caption_alignment":
            raise RegistryInvariantError(
                f"trainer '{trainer_key}' requires cfg.stage='stage2_roi_caption_alignment', got {stage!r}."
            )
        if trainer_key == "stage3_trainer" and stage != "stage3_wsi_report_alignment":
            raise RegistryInvariantError(
                f"trainer '{trainer_key}' requires cfg.stage='stage3_wsi_report_alignment', got {stage!r}."
            )

        batch_size: int = int(cfg.batch_size)
        grad_accum: int = int(cfg.grad_accum_steps)
        effective_batch: int = int(cfg.effective_batch_size)

        if stage == "stage1_titan_v":
            expected: int = _STAGE1_BATCH * _STAGE1_GPUS * _STAGE1_ACCUM
            if batch_size != _STAGE1_BATCH or grad_accum != _STAGE1_ACCUM:
                raise RegistryInvariantError(
                    f"Stage1 requires batch_size={_STAGE1_BATCH}, grad_accum_steps={_STAGE1_ACCUM}; "
                    f"got batch_size={batch_size}, grad_accum_steps={grad_accum}."
                )
            if effective_batch != _STAGE1_EFFECTIVE or expected != effective_batch:
                raise RegistryInvariantError(
                    f"Stage1 effective batch must be {_STAGE1_EFFECTIVE} (computed {expected}), "
                    f"got {effective_batch}."
                )

        if stage == "stage2_roi_caption_alignment":
            expected = _STAGE2_BATCH * _STAGE2_GPUS * _STAGE2_ACCUM
            if batch_size != _STAGE2_BATCH or grad_accum != _STAGE2_ACCUM:
                raise RegistryInvariantError(
                    f"Stage2 requires batch_size={_STAGE2_BATCH}, grad_accum_steps={_STAGE2_ACCUM}; "
                    f"got batch_size={batch_size}, grad_accum_steps={grad_accum}."
                )
            if effective_batch != _STAGE2_EFFECTIVE or expected != effective_batch:
                raise RegistryInvariantError(
                    f"Stage2 effective batch must be {_STAGE2_EFFECTIVE} (computed {expected}), "
                    f"got {effective_batch}."
                )

        if stage == "stage3_wsi_report_alignment":
            expected = _STAGE3_BATCH * _STAGE3_GPUS * _STAGE3_ACCUM
            if batch_size != _STAGE3_BATCH or grad_accum != _STAGE3_ACCUM:
                raise RegistryInvariantError(
                    f"Stage3 requires batch_size={_STAGE3_BATCH}, grad_accum_steps={_STAGE3_ACCUM}; "
                    f"got batch_size={batch_size}, grad_accum_steps={grad_accum}."
                )
            if effective_batch != _STAGE3_EFFECTIVE or expected != effective_batch:
                raise RegistryInvariantError(
                    f"Stage3 effective batch must be {_STAGE3_EFFECTIVE} (computed {expected}), "
                    f"got {effective_batch}."
                )

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _instantiate_with_signature(self, cls: Type[T], kwargs: Mapping[str, Any], context: str) -> T:
        signature = inspect.signature(cls.__init__)
        accepts_var_kwargs: bool = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        filtered_kwargs: Dict[str, Any]
        if accepts_var_kwargs:
            filtered_kwargs = dict(kwargs)
        else:
            valid_names = {
                name
                for name, parameter in signature.parameters.items()
                if name != "self" and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            }
            filtered_kwargs = {name: value for name, value in kwargs.items() if name in valid_names}

        try:
            return cls(**filtered_kwargs)
        except TypeError as exc:
            raise RegistryConstructionError(
                f"Failed constructing {context} using class '{cls.__name__}'. "
                f"Provided kwargs={sorted(filtered_kwargs.keys())}."
            ) from exc

    def _resolve_stage1_groups_path(self, cfg: DataConfig) -> str:
        explicit_groups: Optional[str] = self._safe_nested_get(cfg, ("manifests", "tissue_groups_json"))
        if explicit_groups:
            return explicit_groups

        manifest_path: str = str(cfg.meta_csv)
        parts: Tuple[str, ...] = tuple(manifest_path.replace("\\", "/").split("/"))
        if len(parts) > 1:
            return "/".join(parts[:-1] + ("groups.json",))
        return "groups.json"

    @staticmethod
    def _default_text_encoder_source() -> str:
        return "CONCHv1.5 pretrained text encoder"

    @staticmethod
    def _safe_nested_get(obj: Any, keys: Tuple[str, ...]) -> Optional[Any]:
        current: Any = obj
        for key in keys:
            if current is None:
                return None
            if hasattr(current, key):
                current = getattr(current, key)
                continue
            if isinstance(current, Mapping) and key in current:
                current = current[key]
                continue
            return None
        return current

    @staticmethod
    def _normalize_key(name: str) -> str:
        if not isinstance(name, str):
            raise RegistryRegistrationError("Registry keys must be strings.")
        key: str = name.strip().lower()
        if not key:
            raise RegistryRegistrationError("Registry key cannot be empty.")
        return key

    def _all_dataset_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._dataset_classes.keys()) | set(self._dataset_targets.keys())))

    def _all_model_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._model_classes.keys()) | set(self._model_targets.keys())))

    def _all_trainer_keys(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self._trainer_classes.keys()) | set(self._trainer_targets.keys())))


__all__ = [
    "RegistryError",
    "RegistryRegistrationError",
    "RegistryLookupError",
    "RegistryConstructionError",
    "RegistryInvariantError",
    "Registry",
]
