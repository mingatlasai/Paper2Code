"""Lightning module for THREADS downstream fine-tuning.

This module implements the design-locked public interface:
- ``FinetuneModule.__init__(slide_encoder, head_out_dim, task_type, optim_cfg)``
- ``FinetuneModule.training_step(batch, batch_idx)``
- ``FinetuneModule.validation_step(batch, batch_idx)``
- ``FinetuneModule.predict_step(batch, batch_idx)``

The implementation is aligned to the provided ``config.yaml`` and
``configs/train/finetune.yaml`` contracts:
- THREADS/CHIEF recipe: AdamW, lr=2.5e-5, wd=0.0, no LLRD, no grad accumulation.
- ABMIL recipe compatibility: AdamW, lr=3.0e-4, wd=1.0e-5.
- Classification loss: weighted cross-entropy.
- Survival loss is not specified in config for fine-tuning; survival mode fails
  unless an explicit survival loss spec is provided in ``optim_cfg``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as torch_functional
from torch import nn
from torch.optim import AdamW

from src.models.slide_encoder_threads import ThreadsSlideEncoder


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIM: int = 1024

DEFAULT_TASK_TYPE: str = "binary_classification"
_ALLOWED_TASK_TYPES: Tuple[str, ...] = (
    "binary_classification",
    "subtyping_multiclass",
    "grading_multiclass",
    "survival",
)

DEFAULT_OPTIMIZER_NAME: str = "AdamW"
DEFAULT_THREADS_LEARNING_RATE: float = 2.5e-5
DEFAULT_THREADS_WEIGHT_DECAY: float = 0.0
DEFAULT_ABMIL_LEARNING_RATE: float = 3.0e-4
DEFAULT_ABMIL_WEIGHT_DECAY: float = 1.0e-5
DEFAULT_OPTIMIZER_BETAS: Tuple[float, float] = (0.9, 0.999)
DEFAULT_OPTIMIZER_EPS: float = 1.0e-8

DEFAULT_LAYERWISE_LR_DECAY_ALLOWED: bool = False
DEFAULT_GRADIENT_ACCUMULATION_ALLOWED: bool = False

DEFAULT_LOG_SYNC_DIST: bool = True
DEFAULT_VALIDATE_NUMERICS: bool = True
DEFAULT_METRIC_EPS: float = 1.0e-12

# Batch keys from ``src/data/datamodules.py`` collate.
BATCH_KEY_PATCH_FEATURES: str = "patch_features"
BATCH_KEY_PATCH_MASK: str = "patch_mask"
BATCH_KEY_LABEL: str = "label"
BATCH_KEY_SAMPLE_WEIGHT: str = "sample_weight"
BATCH_KEY_CLASS_WEIGHT: str = "class_weight"
BATCH_KEY_TIME: str = "time"
BATCH_KEY_EVENT: str = "event"
BATCH_KEY_UNIT_ID: str = "unit_id"
BATCH_KEY_SAMPLE_IDS: str = "sample_ids"
BATCH_KEY_PATIENT_ID: str = "patient_id"
BATCH_KEY_TASK_NAME: str = "task_name"


class FinetuneModuleError(Exception):
    """Base exception for finetune module failures."""


class FinetuneModuleConfigError(FinetuneModuleError):
    """Raised when constructor/optimizer/task config is invalid."""


class FinetuneModuleInputError(FinetuneModuleError):
    """Raised when step batch inputs are malformed."""


class FinetuneModuleRuntimeError(FinetuneModuleError):
    """Raised when runtime loss/forward behavior is invalid."""


@dataclass(frozen=True)
class _OptimizerConfig:
    """Resolved optimizer configuration."""

    name: str
    learning_rate: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float
    layerwise_lr_decay: bool
    gradient_accumulation: bool


class FinetuneModule(pl.LightningModule):
    """Lightning module for supervised tuning on slide embeddings.

    The module wraps a pre-built slide encoder and a task head. For classification
    tasks it optimizes weighted cross-entropy. For survival tasks, this module
    requires explicit survival loss configuration in ``optim_cfg`` and otherwise
    fails fast to avoid inventing non-paper behavior.
    """

    def __init__(
        self,
        slide_encoder: ThreadsSlideEncoder,
        head_out_dim: int,
        task_type: str,
        optim_cfg: dict,
    ) -> None:
        """Initialize fine-tuning module.

        Args:
            slide_encoder: THREADS slide encoder returning 1024-d embeddings.
            head_out_dim: Number of output units for task head.
            task_type: Task routing key.
            optim_cfg: Optimizer configuration mapping.
        """
        super().__init__()

        self._slide_encoder: ThreadsSlideEncoder = self._validate_slide_encoder(slide_encoder)
        self._head_out_dim: int = self._validate_head_out_dim(head_out_dim)
        self._task_type: str = self._normalize_task_type(task_type)
        self._optim_cfg_raw: Dict[str, Any] = self._to_dict(optim_cfg)
        self._optimizer_cfg: _OptimizerConfig = self._resolve_optimizer_cfg(self._optim_cfg_raw)

        self._validate_numerics: bool = DEFAULT_VALIDATE_NUMERICS
        self._log_sync_dist: bool = DEFAULT_LOG_SYNC_DIST

        # Survival fine-tuning is unsupported unless explicitly configured.
        self._survival_loss_name: Optional[str] = self._resolve_optional_survival_loss(self._optim_cfg_raw)
        if self._task_type == "survival" and self._survival_loss_name is None:
            raise FinetuneModuleConfigError(
                "task_type='survival' requested, but no explicit survival loss is "
                "configured in optim_cfg. Config-provided fine-tuning only specifies "
                "classification weighted cross-entropy."
            )

        # Task head.
        self._head: nn.Linear = nn.Linear(DEFAULT_EMBEDDING_DIM, self._head_out_dim)
        self._reset_head_parameters()

        # Running diagnostics.
        self._train_loss_sum: float = 0.0
        self._train_step_count: int = 0
        self._val_loss_sum: float = 0.0
        self._val_step_count: int = 0
        self._last_train_metrics: Dict[str, float] = {}
        self._last_val_metrics: Dict[str, float] = {}

        self.save_hyperparameters(
            {
                "task_type": self._task_type,
                "head_out_dim": self._head_out_dim,
                "optimizer": {
                    "name": self._optimizer_cfg.name,
                    "learning_rate": self._optimizer_cfg.learning_rate,
                    "weight_decay": self._optimizer_cfg.weight_decay,
                    "betas": self._optimizer_cfg.betas,
                    "eps": self._optimizer_cfg.eps,
                    "layerwise_lr_decay": self._optimizer_cfg.layerwise_lr_decay,
                    "gradient_accumulation": self._optimizer_cfg.gradient_accumulation,
                },
                "invariants": {
                    "slide_embedding_dim": DEFAULT_EMBEDDING_DIM,
                },
            }
        )

    def configure_optimizers(self) -> object:
        """Configure AdamW optimizer for fine-tuning."""
        parameters: Sequence[nn.Parameter] = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]
        if len(parameters) == 0:
            raise FinetuneModuleConfigError("No trainable parameters found for optimizer.")

        optimizer: AdamW = AdamW(
            params=parameters,
            lr=float(self._optimizer_cfg.learning_rate),
            betas=tuple(self._optimizer_cfg.betas),
            eps=float(self._optimizer_cfg.eps),
            weight_decay=float(self._optimizer_cfg.weight_decay),
        )
        return optimizer

    def training_step(self, batch: dict, batch_idx: int) -> float:
        """Run one supervised training step.

        Args:
            batch: Batch mapping produced by fine-tune datamodule collate.
            batch_idx: Batch index.

        Returns:
            Scalar differentiable loss tensor.
        """
        _ = int(batch_idx)
        loss_tensor, metric_payload = self._compute_supervised_step(batch=batch, train_mode=True)

        loss_value: float = float(loss_tensor.detach().cpu().item())
        self._train_loss_sum += loss_value
        self._train_step_count += 1

        batch_size_value: int = int(self._infer_batch_size(batch))
        self.log(
            "train/loss",
            loss_value,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )
        self.log(
            "train/lr",
            float(self._current_learning_rate()),
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )

        for metric_name, metric_value in metric_payload.items():
            if metric_name in {"train/loss", "train/lr"}:
                continue
            self.log(
                metric_name,
                float(metric_value),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=self._log_sync_dist,
                batch_size=batch_size_value,
            )

        self._last_train_metrics = dict(metric_payload)
        return loss_tensor

    def validation_step(self, batch: dict, batch_idx: int) -> dict[str, float]:
        """Run one validation step.

        Args:
            batch: Batch mapping produced by fine-tune datamodule collate.
            batch_idx: Batch index.

        Returns:
            Dictionary with scalar validation metrics.
        """
        _ = int(batch_idx)
        with torch.no_grad():
            loss_tensor, metric_payload = self._compute_supervised_step(batch=batch, train_mode=False)

        loss_value: float = float(loss_tensor.detach().cpu().item())
        self._val_loss_sum += loss_value
        self._val_step_count += 1

        batch_size_value: int = int(self._infer_batch_size(batch))
        self.log(
            "val/loss",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )

        # Required by configs/train/finetune.yaml logging contract.
        metric_primary: float = float(metric_payload.get("val/metric_primary", float("nan")))
        self.log(
            "val/metric_primary",
            metric_primary,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )

        for metric_name, metric_value in metric_payload.items():
            if metric_name in {"val/loss", "val/metric_primary"}:
                continue
            self.log(
                metric_name,
                float(metric_value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=self._log_sync_dist,
                batch_size=batch_size_value,
            )

        output: Dict[str, float] = {
            "val/loss": loss_value,
            "val/metric_primary": metric_primary,
        }
        for metric_name, metric_value in metric_payload.items():
            if metric_name.startswith("val/") and metric_name not in output:
                output[metric_name] = float(metric_value)

        self._last_val_metrics = dict(output)
        return output

    def predict_step(self, batch: dict, batch_idx: int) -> object:
        """Run deterministic prediction step.

        Args:
            batch: Batch mapping produced by fine-tune datamodule collate.
            batch_idx: Batch index.

        Returns:
            Prediction payload containing logits/scores and identifiers.
        """
        _ = int(batch_idx)
        embeddings: torch.Tensor = self._forward_slide(batch)
        logits_or_risk: torch.Tensor = self._head(embeddings)

        if self._task_type == "survival":
            prediction: torch.Tensor = logits_or_risk.reshape(-1)
        else:
            prediction = logits_or_risk

        payload: Dict[str, Any] = {
            "pred": prediction.detach(),
            "unit_id": batch.get(BATCH_KEY_UNIT_ID),
            "sample_ids": batch.get(BATCH_KEY_SAMPLE_IDS),
            "patient_id": batch.get(BATCH_KEY_PATIENT_ID),
            "task_name": batch.get(BATCH_KEY_TASK_NAME),
            "task_type": self._task_type,
        }

        if self._task_type != "survival":
            payload["logits"] = logits_or_risk.detach()
            payload["proba"] = torch.softmax(logits_or_risk.detach(), dim=1)
        else:
            payload["risk"] = prediction.detach()

        if BATCH_KEY_LABEL in batch:
            payload["label"] = batch[BATCH_KEY_LABEL]
        if BATCH_KEY_TIME in batch:
            payload["time"] = batch[BATCH_KEY_TIME]
        if BATCH_KEY_EVENT in batch:
            payload["event"] = batch[BATCH_KEY_EVENT]

        return payload

    def on_train_epoch_start(self) -> None:
        """Reset epoch-level training accumulators."""
        self._train_loss_sum = 0.0
        self._train_step_count = 0

    def on_train_epoch_end(self) -> None:
        """Log epoch-level aggregate training loss."""
        if self._train_step_count <= 0:
            return
        mean_loss: float = self._train_loss_sum / float(self._train_step_count)
        self.log(
            "train/loss_epoch",
            float(mean_loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self._log_sync_dist,
        )

    def on_validation_epoch_start(self) -> None:
        """Reset epoch-level validation accumulators."""
        self._val_loss_sum = 0.0
        self._val_step_count = 0

    def on_validation_epoch_end(self) -> None:
        """Log epoch-level aggregate validation loss."""
        if self._val_step_count <= 0:
            return
        mean_loss: float = self._val_loss_sum / float(self._val_step_count)
        self.log(
            "val/loss_epoch",
            float(mean_loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=self._log_sync_dist,
        )

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------
    def _compute_supervised_step(
        self,
        *,
        batch: Mapping[str, Any],
        train_mode: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not isinstance(batch, Mapping):
            raise FinetuneModuleInputError(
                f"batch must be mapping, got {type(batch).__name__}."
            )

        embeddings: torch.Tensor = self._forward_slide(batch)
        logits: torch.Tensor = self._head(embeddings)

        if self._task_type == "survival":
            return self._compute_survival_step(
                batch=batch,
                logits_or_risk=logits,
                train_mode=train_mode,
            )

        return self._compute_classification_step(
            batch=batch,
            logits=logits,
            train_mode=train_mode,
        )

    def _forward_slide(self, batch: Mapping[str, Any]) -> torch.Tensor:
        patch_features: Any = self._require_key(batch, BATCH_KEY_PATCH_FEATURES)
        patch_mask: Any = self._require_key(batch, BATCH_KEY_PATCH_MASK)

        embeddings_any: Any = self._slide_encoder(
            patch_features=patch_features,
            patch_mask=patch_mask,
        )
        embeddings: torch.Tensor = self._coerce_tensor(embeddings_any, name="slide_embedding")

        if embeddings.ndim != 2:
            raise FinetuneModuleRuntimeError(
                f"slide encoder output must be rank-2 [B,D], got {tuple(embeddings.shape)}."
            )
        if int(embeddings.shape[1]) != DEFAULT_EMBEDDING_DIM:
            raise FinetuneModuleRuntimeError(
                "slide encoder embedding dim mismatch: "
                f"expected {DEFAULT_EMBEDDING_DIM}, got {int(embeddings.shape[1])}."
            )
        if self._validate_numerics and not torch.isfinite(embeddings).all():
            raise FinetuneModuleRuntimeError("slide embeddings contain NaN/Inf.")

        return embeddings

    def _compute_classification_step(
        self,
        *,
        batch: Mapping[str, Any],
        logits: torch.Tensor,
        train_mode: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        labels_any: Any = self._require_key(batch, BATCH_KEY_LABEL)
        labels: torch.Tensor = self._coerce_labels(labels_any)

        if logits.ndim != 2:
            raise FinetuneModuleInputError(
                f"classification logits must be rank-2 [B,C], got {tuple(logits.shape)}."
            )
        if int(logits.shape[0]) != int(labels.shape[0]):
            raise FinetuneModuleInputError(
                f"Batch size mismatch between logits and labels: {int(logits.shape[0])} vs {int(labels.shape[0])}."
            )

        class_weight: Optional[torch.Tensor] = self._resolve_class_weight(batch, logits)
        sample_weight: Optional[torch.Tensor] = self._resolve_sample_weight(batch, logits)

        if sample_weight is None:
            loss_tensor: torch.Tensor = torch_functional.cross_entropy(
                logits,
                labels,
                weight=class_weight,
            )
        else:
            per_sample: torch.Tensor = torch_functional.cross_entropy(
                logits,
                labels,
                weight=class_weight,
                reduction="none",
            )
            weighted_sum: torch.Tensor = torch.sum(per_sample * sample_weight)
            normalizer: torch.Tensor = torch.sum(sample_weight).clamp_min(DEFAULT_METRIC_EPS)
            loss_tensor = weighted_sum / normalizer

        self._validate_loss_tensor(loss_tensor, name="classification_loss")

        predictions: torch.Tensor = torch.argmax(logits.detach(), dim=1)
        accuracy_value: float = float((predictions == labels).to(torch.float32).mean().cpu().item())

        prefix: str = "train" if train_mode else "val"
        metrics: Dict[str, float] = {
            f"{prefix}/loss": float(loss_tensor.detach().cpu().item()),
            f"{prefix}/metric_primary": accuracy_value,
            f"{prefix}/accuracy": accuracy_value,
            f"{prefix}/logit_norm": float(torch.norm(logits.detach(), p=2).cpu().item()),
        }

        # Optional binary positive-rate signal.
        if int(logits.shape[1]) == 2:
            positive_rate: float = float((labels == 1).to(torch.float32).mean().cpu().item())
            metrics[f"{prefix}/positive_rate_batch"] = positive_rate

        return loss_tensor, metrics

    def _compute_survival_step(
        self,
        *,
        batch: Mapping[str, Any],
        logits_or_risk: torch.Tensor,
        train_mode: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if self._survival_loss_name is None:
            raise FinetuneModuleConfigError(
                "Survival step reached without configured survival loss."
            )

        time_any: Any = self._require_key(batch, BATCH_KEY_TIME)
        event_any: Any = self._require_key(batch, BATCH_KEY_EVENT)

        time_tensor: torch.Tensor = self._coerce_tensor(time_any, name=BATCH_KEY_TIME).to(torch.float32)
        event_tensor: torch.Tensor = self._coerce_tensor(event_any, name=BATCH_KEY_EVENT).to(torch.float32)

        risk: torch.Tensor = logits_or_risk.reshape(-1)
        if time_tensor.ndim != 1:
            time_tensor = time_tensor.reshape(-1)
        if event_tensor.ndim != 1:
            event_tensor = event_tensor.reshape(-1)

        if int(risk.shape[0]) != int(time_tensor.shape[0]) or int(risk.shape[0]) != int(event_tensor.shape[0]):
            raise FinetuneModuleInputError(
                "Survival tensors must share batch size: "
                f"risk={tuple(risk.shape)}, time={tuple(time_tensor.shape)}, event={tuple(event_tensor.shape)}."
            )

        # No paper-provided survival fine-tuning loss in config; explicit fallback only when requested.
        survival_loss_name: str = self._survival_loss_name.strip().lower()
        if survival_loss_name in {"mse", "mse_loss", "regression_mse"}:
            # Event-weighted regression surrogate (explicitly non-paper, opt-in only).
            weight: torch.Tensor = torch.where(event_tensor > 0.0, torch.ones_like(event_tensor), 0.5 * torch.ones_like(event_tensor))
            mse_per_sample: torch.Tensor = (risk - time_tensor) ** 2
            loss_tensor: torch.Tensor = torch.sum(mse_per_sample * weight) / torch.sum(weight).clamp_min(DEFAULT_METRIC_EPS)
        else:
            raise FinetuneModuleConfigError(
                "Unsupported explicit survival loss for fine-tuning: "
                f"{self._survival_loss_name!r}. Supported opt-in: 'mse'."
            )

        self._validate_loss_tensor(loss_tensor, name="survival_loss")

        prefix: str = "train" if train_mode else "val"
        metrics: Dict[str, float] = {
            f"{prefix}/loss": float(loss_tensor.detach().cpu().item()),
            f"{prefix}/metric_primary": float(-loss_tensor.detach().cpu().item()),
            f"{prefix}/risk_mean": float(risk.detach().mean().cpu().item()),
        }
        return loss_tensor, metrics

    # ------------------------------------------------------------------
    # Parsing and validation
    # ------------------------------------------------------------------
    def _resolve_optimizer_cfg(self, raw_cfg: Mapping[str, Any]) -> _OptimizerConfig:
        name: str = self._as_str(raw_cfg.get("name"), key="optimizer.name", default=DEFAULT_OPTIMIZER_NAME)
        if name.strip().lower() != DEFAULT_OPTIMIZER_NAME.lower():
            raise FinetuneModuleConfigError(
                f"optimizer.name must be '{DEFAULT_OPTIMIZER_NAME}', got {name!r}."
            )

        learning_rate: float = self._as_float(
            raw_cfg.get("learning_rate", raw_cfg.get("lr")),
            key="optimizer.learning_rate",
            default=DEFAULT_THREADS_LEARNING_RATE,
        )
        weight_decay: float = self._as_float(
            raw_cfg.get("weight_decay"),
            key="optimizer.weight_decay",
            default=DEFAULT_THREADS_WEIGHT_DECAY,
        )

        # Recipe compatibility: THREADS/CHIEF or ABMIL.
        threads_recipe: bool = math.isclose(learning_rate, DEFAULT_THREADS_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-16)
        abmil_recipe: bool = math.isclose(learning_rate, DEFAULT_ABMIL_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-16)
        if threads_recipe:
            if not math.isclose(weight_decay, DEFAULT_THREADS_WEIGHT_DECAY, rel_tol=0.0, abs_tol=1e-16):
                raise FinetuneModuleConfigError(
                    "THREADS/CHIEF optimizer weight_decay invariant violated: "
                    f"expected {DEFAULT_THREADS_WEIGHT_DECAY}, got {weight_decay}."
                )
        elif abmil_recipe:
            if not math.isclose(weight_decay, DEFAULT_ABMIL_WEIGHT_DECAY, rel_tol=0.0, abs_tol=1e-16):
                raise FinetuneModuleConfigError(
                    "ABMIL optimizer weight_decay invariant violated: "
                    f"expected {DEFAULT_ABMIL_WEIGHT_DECAY}, got {weight_decay}."
                )
        else:
            raise FinetuneModuleConfigError(
                "optimizer.learning_rate must match a configured recipe. "
                f"Allowed: {DEFAULT_THREADS_LEARNING_RATE} (THREADS/CHIEF) or "
                f"{DEFAULT_ABMIL_LEARNING_RATE} (ABMIL); got {learning_rate}."
            )

        layerwise_lr_decay: bool = self._as_bool(
            raw_cfg.get("layerwise_lr_decay", False),
            key="optimizer.layerwise_lr_decay",
            default=False,
        )
        gradient_accumulation: bool = self._as_bool(
            raw_cfg.get("gradient_accumulation", False),
            key="optimizer.gradient_accumulation",
            default=False,
        )
        if layerwise_lr_decay != DEFAULT_LAYERWISE_LR_DECAY_ALLOWED:
            raise FinetuneModuleConfigError(
                "layerwise_lr_decay is not allowed in this module per config contract."
            )
        if gradient_accumulation != DEFAULT_GRADIENT_ACCUMULATION_ALLOWED:
            raise FinetuneModuleConfigError(
                "gradient_accumulation is not allowed in this module per config contract."
            )

        betas: Tuple[float, float] = self._parse_betas(raw_cfg.get("betas"))
        eps: float = self._as_float(
            raw_cfg.get("eps"),
            key="optimizer.eps",
            default=DEFAULT_OPTIMIZER_EPS,
        )

        return _OptimizerConfig(
            name=DEFAULT_OPTIMIZER_NAME,
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            layerwise_lr_decay=bool(layerwise_lr_decay),
            gradient_accumulation=bool(gradient_accumulation),
        )

    def _resolve_optional_survival_loss(self, raw_cfg: Mapping[str, Any]) -> Optional[str]:
        if "loss" in raw_cfg and isinstance(raw_cfg["loss"], Mapping):
            loss_mapping: Mapping[str, Any] = raw_cfg["loss"]
            if "survival" in loss_mapping and loss_mapping["survival"] is not None:
                loss_name: str = str(loss_mapping["survival"]).strip()
                return loss_name if loss_name != "" else None
        if "survival_loss" in raw_cfg and raw_cfg["survival_loss"] is not None:
            loss_name = str(raw_cfg["survival_loss"]).strip()
            return loss_name if loss_name != "" else None
        return None

    def _resolve_class_weight(self, batch: Mapping[str, Any], logits: torch.Tensor) -> Optional[torch.Tensor]:
        if BATCH_KEY_CLASS_WEIGHT not in batch:
            return None

        class_weight_tensor: torch.Tensor = self._coerce_tensor(
            batch[BATCH_KEY_CLASS_WEIGHT],
            name=BATCH_KEY_CLASS_WEIGHT,
        ).to(dtype=torch.float32, device=logits.device)

        if class_weight_tensor.ndim != 1:
            raise FinetuneModuleInputError(
                f"{BATCH_KEY_CLASS_WEIGHT} must be rank-1 [C], got {tuple(class_weight_tensor.shape)}."
            )
        if int(class_weight_tensor.shape[0]) != int(logits.shape[1]):
            raise FinetuneModuleInputError(
                f"{BATCH_KEY_CLASS_WEIGHT} length mismatch: expected {int(logits.shape[1])}, got {int(class_weight_tensor.shape[0])}."
            )
        if self._validate_numerics and not torch.isfinite(class_weight_tensor).all():
            raise FinetuneModuleInputError(f"{BATCH_KEY_CLASS_WEIGHT} contains NaN/Inf.")
        return class_weight_tensor

    def _resolve_sample_weight(self, batch: Mapping[str, Any], logits: torch.Tensor) -> Optional[torch.Tensor]:
        if BATCH_KEY_SAMPLE_WEIGHT not in batch:
            return None

        sample_weight_tensor: torch.Tensor = self._coerce_tensor(
            batch[BATCH_KEY_SAMPLE_WEIGHT],
            name=BATCH_KEY_SAMPLE_WEIGHT,
        ).to(dtype=torch.float32, device=logits.device)

        if sample_weight_tensor.ndim != 1:
            sample_weight_tensor = sample_weight_tensor.reshape(-1)

        if int(sample_weight_tensor.shape[0]) != int(logits.shape[0]):
            raise FinetuneModuleInputError(
                f"{BATCH_KEY_SAMPLE_WEIGHT} length mismatch: expected {int(logits.shape[0])}, got {int(sample_weight_tensor.shape[0])}."
            )
        if self._validate_numerics and not torch.isfinite(sample_weight_tensor).all():
            raise FinetuneModuleInputError(f"{BATCH_KEY_SAMPLE_WEIGHT} contains NaN/Inf.")
        return sample_weight_tensor

    def _coerce_labels(self, labels: Any) -> torch.Tensor:
        labels_tensor: torch.Tensor = self._coerce_tensor(labels, name=BATCH_KEY_LABEL)
        if labels_tensor.ndim != 1:
            labels_tensor = labels_tensor.reshape(-1)
        labels_tensor = labels_tensor.to(torch.long)
        if (labels_tensor < 0).any():
            raise FinetuneModuleInputError("Labels must be non-negative class indices.")
        return labels_tensor

    def _coerce_tensor(self, value: Any, name: str) -> torch.Tensor:
        try:
            tensor_value: torch.Tensor = torch.as_tensor(value)
        except Exception as exc:  # noqa: BLE001
            raise FinetuneModuleInputError(
                f"{name} cannot be converted to tensor: {exc}"
            ) from exc

        if torch.is_floating_point(tensor_value):
            tensor_value = tensor_value.to(torch.float32)
            if self._validate_numerics and not torch.isfinite(tensor_value).all():
                raise FinetuneModuleInputError(f"{name} contains NaN/Inf values.")

        return tensor_value

    def _validate_loss_tensor(self, tensor_value: torch.Tensor, name: str) -> None:
        if not isinstance(tensor_value, torch.Tensor):
            raise FinetuneModuleRuntimeError(f"{name} must be torch.Tensor.")
        if tensor_value.ndim > 1:
            raise FinetuneModuleRuntimeError(
                f"{name} must be scalar-like, got shape={tuple(tensor_value.shape)}."
            )
        if self._validate_numerics and not torch.isfinite(tensor_value):
            raise FinetuneModuleRuntimeError(f"{name} is non-finite.")

    def _current_learning_rate(self) -> float:
        try:
            optimizer_value: Any = self.optimizers()
        except Exception:
            optimizer_value = None

        if optimizer_value is None:
            return float(self._optimizer_cfg.learning_rate)

        if isinstance(optimizer_value, (list, tuple)):
            if len(optimizer_value) == 0:
                return float(self._optimizer_cfg.learning_rate)
            optimizer_value = optimizer_value[0]

        if not hasattr(optimizer_value, "param_groups"):
            return float(self._optimizer_cfg.learning_rate)

        param_groups: Any = getattr(optimizer_value, "param_groups")
        if not isinstance(param_groups, list) or len(param_groups) == 0:
            return float(self._optimizer_cfg.learning_rate)

        return float(param_groups[0].get("lr", self._optimizer_cfg.learning_rate))

    def _infer_batch_size(self, batch: Mapping[str, Any]) -> int:
        if BATCH_KEY_PATCH_FEATURES not in batch:
            return 1
        patch_tensor: torch.Tensor = self._coerce_tensor(batch[BATCH_KEY_PATCH_FEATURES], name=BATCH_KEY_PATCH_FEATURES)
        if patch_tensor.ndim >= 3:
            return int(patch_tensor.shape[0])
        return 1

    @staticmethod
    def _validate_slide_encoder(slide_encoder: ThreadsSlideEncoder) -> ThreadsSlideEncoder:
        if not isinstance(slide_encoder, ThreadsSlideEncoder):
            raise FinetuneModuleConfigError(
                f"slide_encoder must be ThreadsSlideEncoder, got {type(slide_encoder).__name__}."
            )

        out_dim: Optional[int] = getattr(slide_encoder, "_out_dim", None)
        if out_dim is not None and int(out_dim) != DEFAULT_EMBEDDING_DIM:
            raise FinetuneModuleConfigError(
                f"slide_encoder output dim mismatch: expected {DEFAULT_EMBEDDING_DIM}, got {out_dim}."
            )
        return slide_encoder

    @staticmethod
    def _validate_head_out_dim(head_out_dim: int) -> int:
        if isinstance(head_out_dim, bool) or not isinstance(head_out_dim, int):
            raise FinetuneModuleConfigError(
                f"head_out_dim must be int, got {type(head_out_dim).__name__}."
            )
        if head_out_dim <= 0:
            raise FinetuneModuleConfigError(
                f"head_out_dim must be > 0, got {head_out_dim}."
            )
        return int(head_out_dim)

    @staticmethod
    def _normalize_task_type(task_type: str) -> str:
        normalized: str = str(task_type).strip().lower() if task_type is not None else ""
        if normalized == "":
            normalized = DEFAULT_TASK_TYPE
        if normalized not in _ALLOWED_TASK_TYPES:
            raise FinetuneModuleConfigError(
                f"Unsupported task_type={task_type!r}. Allowed: {_ALLOWED_TASK_TYPES}."
            )
        return normalized

    def _reset_head_parameters(self) -> None:
        nn.init.xavier_uniform_(self._head.weight)
        if self._head.bias is not None:
            nn.init.zeros_(self._head.bias)

    @staticmethod
    def _require_key(mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise FinetuneModuleInputError(f"Missing required batch key: {key!r}")
        return mapping[key]

    @staticmethod
    def _to_dict(value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return dict(value)
        if isinstance(value, Mapping):
            return dict(value)
        if hasattr(value, "to_dict") and callable(value.to_dict):
            converted: Any = value.to_dict()
            if isinstance(converted, Mapping):
                return dict(converted)
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(value):
                container: Any = OmegaConf.to_container(value, resolve=True)
                if isinstance(container, Mapping):
                    return dict(container)
        except Exception:
            pass
        raise FinetuneModuleConfigError(
            f"Unsupported config object type: {type(value).__name__}."
        )

    @staticmethod
    def _as_str(value: Any, *, key: str, default: str) -> str:
        if value is None:
            return default
        if isinstance(value, str):
            normalized: str = value.strip()
            return normalized if normalized != "" else default
        return str(value)

    @staticmethod
    def _as_int(value: Any, *, key: str, default: int) -> int:
        if value is None:
            return int(default)
        if isinstance(value, bool):
            raise FinetuneModuleConfigError(f"{key} must be int, got bool.")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise FinetuneModuleConfigError(
                    f"{key} must be integer-like, got {value}."
                )
            return int(value)
        if isinstance(value, str):
            stripped: str = value.strip()
            if stripped == "":
                return int(default)
            try:
                return int(stripped)
            except ValueError as exc:
                raise FinetuneModuleConfigError(f"{key} must be int, got {value!r}.") from exc
        raise FinetuneModuleConfigError(f"{key} must be int, got {type(value).__name__}.")

    @staticmethod
    def _as_float(value: Any, *, key: str, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            raise FinetuneModuleConfigError(f"{key} must be float, got bool.")
        if isinstance(value, (int, float)):
            parsed: float = float(value)
            if not math.isfinite(parsed):
                raise FinetuneModuleConfigError(f"{key} must be finite, got {parsed}.")
            return parsed
        if isinstance(value, str):
            stripped: str = value.strip()
            if stripped == "":
                return float(default)
            try:
                parsed = float(stripped)
            except ValueError as exc:
                raise FinetuneModuleConfigError(
                    f"{key} must be float, got {value!r}."
                ) from exc
            if not math.isfinite(parsed):
                raise FinetuneModuleConfigError(f"{key} must be finite, got {parsed}.")
            return parsed
        raise FinetuneModuleConfigError(f"{key} must be float, got {type(value).__name__}.")

    @staticmethod
    def _as_bool(value: Any, *, key: str, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized: str = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off"}:
                return False
        raise FinetuneModuleConfigError(f"{key} must be bool, got {value!r}.")

    @staticmethod
    def _parse_betas(value: Any) -> Tuple[float, float]:
        if value is None:
            return DEFAULT_OPTIMIZER_BETAS
        if isinstance(value, (list, tuple)) and len(value) == 2:
            beta1: float = float(value[0])
            beta2: float = float(value[1])
            if not (0.0 < beta1 < 1.0 and 0.0 < beta2 < 1.0):
                raise FinetuneModuleConfigError(
                    f"optimizer.betas must be in (0,1), got {(beta1, beta2)}."
                )
            return beta1, beta2
        raise FinetuneModuleConfigError(
            f"optimizer.betas must be null or length-2 sequence, got {value!r}."
        )


__all__ = [
    "FinetuneModuleError",
    "FinetuneModuleConfigError",
    "FinetuneModuleInputError",
    "FinetuneModuleRuntimeError",
    "FinetuneModule",
]
