"""Lightning module for THREADS multimodal pretraining.

This module implements the design-locked interface:
- ``PretrainModule.__init__(model: ThreadsModel, optim_cfg: dict, sched_cfg: dict) -> None``
- ``PretrainModule.configure_optimizers() -> object``
- ``PretrainModule.training_step(batch: dict, batch_idx: int) -> float``
- ``PretrainModule.validation_step(batch: dict, batch_idx: int) -> dict[str, float]``

Implementation notes:
- Uses AdamW with warmup + cosine decay, strictly config-driven.
- Computes differentiable loss directly from ``ThreadsModel.forward_*`` branches,
  because ``ThreadsModel.training_step`` is diagnostic-only and returns floats.
- Exposes epoch slide embeddings for RankMe callback compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.models.threads_model import ThreadsModel


# -----------------------------------------------------------------------------
# Config defaults (paper/config aligned)
# -----------------------------------------------------------------------------
DEFAULT_OPTIMIZER_NAME: str = "AdamW"
DEFAULT_LEARNING_RATE: float = 1.0e-5
DEFAULT_BETAS: Tuple[float, float] = (0.9, 0.999)
DEFAULT_EPS: float = 1.0e-8
DEFAULT_WEIGHT_DECAY: float = 0.0

DEFAULT_SCHEDULER_TYPE: str = "cosine decay"
DEFAULT_MAX_EPOCHS: int = 101
DEFAULT_WARMUP_EPOCHS: int = 5
DEFAULT_WARMUP_START_LR: float = 0.0
DEFAULT_PEAK_LR: float = 1.0e-5
DEFAULT_FINAL_LR: float = 0.0

DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_DNA_INPUT_DIM: int = 1673

DEFAULT_MODALITY_WEIGHT_RNA: float = 1.0
DEFAULT_MODALITY_WEIGHT_DNA: float = 1.0
DEFAULT_SKIP_MISSING_MODALITY: bool = True
DEFAULT_FAIL_IF_NO_VALID_PAIR: bool = True

DEFAULT_STRICT_CONFIG: bool = True
DEFAULT_LOG_SYNC_DIST: bool = True
DEFAULT_VALIDATE_NUMERICS: bool = True
DEFAULT_METRIC_EPS: float = 1.0e-12

# Batch keys expected from datamodule collate.
BATCH_KEY_PATCH_FEATURES: str = "patch_features"
BATCH_KEY_PATCH_MASK: str = "patch_mask"
BATCH_KEY_SAMPLE_ID: str = "sample_id"

BATCH_KEY_HAS_RNA: str = "has_rna"
BATCH_KEY_GENE_IDS: str = "gene_ids"
BATCH_KEY_EXPR_VALS: str = "expr_vals"
BATCH_KEY_GENE_MASK: str = "gene_mask"

BATCH_KEY_HAS_DNA: str = "has_dna"
BATCH_KEY_DNA_MULTI_HOT: str = "dna_multi_hot"


class PretrainModuleError(Exception):
    """Base exception for pretrain module failures."""


class PretrainModuleConfigError(PretrainModuleError):
    """Raised when optimizer/scheduler/model config is invalid."""


class PretrainModuleInputError(PretrainModuleError):
    """Raised when batch input is malformed."""


class PretrainModuleRuntimeError(PretrainModuleError):
    """Raised when runtime loss/logging behavior is invalid."""


@dataclass(frozen=True)
class _OptimizerConfig:
    """Resolved optimizer configuration."""

    name: str
    learning_rate: float
    betas: Tuple[float, float]
    eps: float
    weight_decay: float
    zero_grad_set_to_none: bool


@dataclass(frozen=True)
class _SchedulerConfig:
    """Resolved scheduler configuration."""

    scheduler_type: str
    max_epochs: int
    warmup_enabled: bool
    warmup_epochs: int
    warmup_start_lr: float
    peak_lr: float
    final_lr: float


class PretrainModule(pl.LightningModule):
    """Lightning module for THREADS multimodal contrastive pretraining."""

    def __init__(self, model: ThreadsModel, optim_cfg: dict, sched_cfg: dict) -> None:
        """Initialize pretraining Lightning module.

        Args:
            model: Composed THREADS model.
            optim_cfg: Optimizer config mapping.
            sched_cfg: Scheduler config mapping.
        """
        super().__init__()

        self._model: ThreadsModel = self._validate_model(model)
        self._optim_cfg_raw: Dict[str, Any] = self._to_dict(optim_cfg)
        self._sched_cfg_raw: Dict[str, Any] = self._to_dict(sched_cfg)

        self._strict_config: bool = DEFAULT_STRICT_CONFIG
        self._validate_numerics: bool = DEFAULT_VALIDATE_NUMERICS
        self._log_sync_dist: bool = DEFAULT_LOG_SYNC_DIST

        self._optimizer_config: _OptimizerConfig = self._resolve_optimizer_config(
            self._optim_cfg_raw
        )
        self._scheduler_config: _SchedulerConfig = self._resolve_scheduler_config(
            self._sched_cfg_raw,
            optimizer_lr=self._optimizer_config.learning_rate,
        )

        self._weight_rna: float = self._resolve_modality_weight(
            self._optim_cfg_raw,
            modality="wsi_rna",
            default=DEFAULT_MODALITY_WEIGHT_RNA,
        )
        self._weight_dna: float = self._resolve_modality_weight(
            self._optim_cfg_raw,
            modality="wsi_dna",
            default=DEFAULT_MODALITY_WEIGHT_DNA,
        )
        self._skip_missing_modality: bool = self._resolve_bool(
            self._optim_cfg_raw,
            paths=(("skip_missing_modality",),),
            default=DEFAULT_SKIP_MISSING_MODALITY,
        )
        self._fail_if_no_valid_pair: bool = self._resolve_bool(
            self._optim_cfg_raw,
            paths=(("fail_if_no_valid_pair_in_batch",),),
            default=DEFAULT_FAIL_IF_NO_VALID_PAIR,
        )

        # Epoch aggregation state.
        self._train_loss_sum: float = 0.0
        self._train_step_count: int = 0
        self._val_loss_sum: float = 0.0
        self._val_step_count: int = 0

        # RankMe callback compatibility payload.
        self.rankme_embeddings: Optional[torch.Tensor] = None
        self._rankme_embeddings: Optional[torch.Tensor] = None
        self._epoch_slide_embeddings: List[torch.Tensor] = []

        # Keep last metrics for pipeline/checkpoint integration.
        self._last_train_metrics: Dict[str, float] = {}
        self._last_val_metrics: Dict[str, float] = {}

        # Expose warmup boundary metadata for callback/debugging.
        self.rankme_warmup_epochs: int = int(self._scheduler_config.warmup_epochs)
        self.rankme_start_epoch: int = int(self._scheduler_config.warmup_epochs + 1)

        # Lightning will serialize these via checkpoint; useful for reproducibility.
        self.save_hyperparameters(
            {
                "optimizer": {
                    "name": self._optimizer_config.name,
                    "learning_rate": self._optimizer_config.learning_rate,
                    "betas": self._optimizer_config.betas,
                    "eps": self._optimizer_config.eps,
                    "weight_decay": self._optimizer_config.weight_decay,
                    "zero_grad_set_to_none": self._optimizer_config.zero_grad_set_to_none,
                },
                "scheduler": {
                    "type": self._scheduler_config.scheduler_type,
                    "max_epochs": self._scheduler_config.max_epochs,
                    "warmup_enabled": self._scheduler_config.warmup_enabled,
                    "warmup_epochs": self._scheduler_config.warmup_epochs,
                    "warmup_start_lr": self._scheduler_config.warmup_start_lr,
                    "peak_lr": self._scheduler_config.peak_lr,
                    "final_lr": self._scheduler_config.final_lr,
                },
                "loss_weights": {
                    "wsi_rna": self._weight_rna,
                    "wsi_dna": self._weight_dna,
                },
                "flags": {
                    "skip_missing_modality": self._skip_missing_modality,
                    "fail_if_no_valid_pair_in_batch": self._fail_if_no_valid_pair,
                },
                "invariants": {
                    "embedding_dim": DEFAULT_EMBEDDING_DIM,
                    "dna_input_dim": DEFAULT_DNA_INPUT_DIM,
                },
            }
        )

    @property
    def model(self) -> ThreadsModel:
        """Return wrapped THREADS model."""
        return self._model

    def configure_optimizers(self) -> object:
        """Configure AdamW optimizer and warmup+cosine scheduler."""
        parameters: List[torch.nn.Parameter] = [
            parameter for parameter in self._model.parameters() if parameter.requires_grad
        ]
        if len(parameters) == 0:
            raise PretrainModuleConfigError("No trainable parameters found for optimizer.")

        optimizer: AdamW = AdamW(
            params=parameters,
            lr=float(self._optimizer_config.learning_rate),
            betas=tuple(self._optimizer_config.betas),
            eps=float(self._optimizer_config.eps),
            weight_decay=float(self._optimizer_config.weight_decay),
        )

        scheduler: LambdaLR = LambdaLR(
            optimizer=optimizer,
            lr_lambda=self._build_lr_lambda(),
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "lr/warmup_cosine",
            },
        }

    def training_step(self, batch: dict, batch_idx: int) -> float:
        """Compute one training step loss and log metrics.

        Args:
            batch: Pretraining batch mapping from datamodule.
            batch_idx: Lightning batch index.

        Returns:
            Scalar differentiable loss tensor.
        """
        _ = int(batch_idx)
        loss_tensor, metric_payload, slide_embeddings = self._compute_loss_from_batch(
            batch=batch,
            train_mode=True,
        )

        # Keep embeddings for RankMe callback at epoch-end.
        self._epoch_slide_embeddings.append(slide_embeddings.detach().cpu())

        # Optional diagnostics from ThreadsModel.training_step (float metrics).
        diagnostic_payload: Dict[str, float] = self._safe_model_diagnostics(batch)
        merged_metrics: Dict[str, float] = dict(metric_payload)
        merged_metrics.update(diagnostic_payload)

        self._train_loss_sum += float(loss_tensor.detach().cpu().item())
        self._train_step_count += 1

        self._log_train_step_metrics(merged_metrics)
        self._last_train_metrics = dict(merged_metrics)

        return loss_tensor

    def validation_step(self, batch: dict, batch_idx: int) -> dict[str, float]:
        """Compute one validation step and return scalar metric dictionary."""
        _ = int(batch_idx)

        with torch.no_grad():
            loss_tensor, metric_payload, _slide_embeddings = self._compute_loss_from_batch(
                batch=batch,
                train_mode=False,
            )

        diagnostic_payload: Dict[str, float] = self._safe_model_diagnostics(batch)
        merged_metrics: Dict[str, float] = dict(metric_payload)
        merged_metrics.update({f"val/{key.split('/', 1)[-1]}": value for key, value in diagnostic_payload.items() if key.startswith("train/")})

        loss_value: float = float(loss_tensor.detach().cpu().item())
        self._val_loss_sum += loss_value
        self._val_step_count += 1

        self.log(
            "val/loss_total",
            loss_value,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=int(self._infer_batch_size(batch)),
        )

        for metric_name, metric_value in merged_metrics.items():
            if metric_name == "val/loss_total":
                continue
            if not metric_name.startswith("val/"):
                continue
            self.log(
                metric_name,
                float(metric_value),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                sync_dist=self._log_sync_dist,
                batch_size=int(self._infer_batch_size(batch)),
            )

        self._last_val_metrics = dict(merged_metrics)
        output: Dict[str, float] = {"val/loss_total": loss_value}
        output.update({key: float(value) for key, value in merged_metrics.items() if key.startswith("val/")})
        return output

    def on_train_epoch_start(self) -> None:
        """Reset epoch-level train aggregators."""
        self._train_loss_sum = 0.0
        self._train_step_count = 0
        self._epoch_slide_embeddings = []
        self.rankme_embeddings = None
        self._rankme_embeddings = None

    def on_train_epoch_end(self) -> None:
        """Finalize train epoch metrics and expose RankMe embeddings."""
        if self._train_step_count > 0:
            mean_train_loss: float = self._train_loss_sum / float(self._train_step_count)
            self.log(
                "train/loss_total_epoch",
                float(mean_train_loss),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=self._log_sync_dist,
            )

        if len(self._epoch_slide_embeddings) > 0:
            rankme_tensor: torch.Tensor = torch.cat(self._epoch_slide_embeddings, dim=0)
            self.rankme_embeddings = rankme_tensor
            self._rankme_embeddings = rankme_tensor
        else:
            self.rankme_embeddings = None
            self._rankme_embeddings = None

    def on_validation_epoch_start(self) -> None:
        """Reset epoch-level validation aggregators."""
        self._val_loss_sum = 0.0
        self._val_step_count = 0

    def on_validation_epoch_end(self) -> None:
        """Finalize validation epoch metrics."""
        if self._val_step_count > 0:
            mean_val_loss: float = self._val_loss_sum / float(self._val_step_count)
            self.log(
                "val/loss_total_epoch",
                float(mean_val_loss),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=self._log_sync_dist,
            )

    # ------------------------------------------------------------------
    # Internal loss path
    # ------------------------------------------------------------------
    def _compute_loss_from_batch(
        self,
        *,
        batch: Mapping[str, Any],
        train_mode: bool,
    ) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
        if not isinstance(batch, Mapping):
            raise PretrainModuleInputError(
                f"batch must be mapping, got {type(batch).__name__}."
            )

        patch_features: object = self._require_key(batch, BATCH_KEY_PATCH_FEATURES)
        patch_mask: object = self._require_key(batch, BATCH_KEY_PATCH_MASK)

        z_wsi_any: Any = self._model.forward_wsi(patches=patch_features, patch_mask=patch_mask)
        z_wsi: torch.Tensor = self._coerce_tensor(z_wsi_any, name="z_wsi")
        self._validate_embedding_tensor(z_wsi, name="z_wsi")

        batch_size: int = int(z_wsi.shape[0])

        loss_terms: List[torch.Tensor] = []
        loss_weights: List[float] = []

        metrics: Dict[str, float] = {
            "train/n_pairs_rna": 0.0,
            "train/n_pairs_dna": 0.0,
        }

        # RNA branch.
        rna_indices: torch.Tensor = self._resolve_modality_indices(
            batch=batch,
            modality_flag_key=BATCH_KEY_HAS_RNA,
            batch_size=batch_size,
        )
        if int(rna_indices.numel()) > 0:
            gene_ids: torch.Tensor = self._coerce_tensor(
                self._require_key(batch, BATCH_KEY_GENE_IDS),
                name=BATCH_KEY_GENE_IDS,
            )
            expr_vals: torch.Tensor = self._coerce_tensor(
                self._require_key(batch, BATCH_KEY_EXPR_VALS),
                name=BATCH_KEY_EXPR_VALS,
            )
            gene_mask: torch.Tensor = self._coerce_tensor(
                self._require_key(batch, BATCH_KEY_GENE_MASK),
                name=BATCH_KEY_GENE_MASK,
            )

            self._validate_first_dim(gene_ids, batch_size, BATCH_KEY_GENE_IDS)
            self._validate_first_dim(expr_vals, batch_size, BATCH_KEY_EXPR_VALS)
            self._validate_first_dim(gene_mask, batch_size, BATCH_KEY_GENE_MASK)

            z_wsi_rna: torch.Tensor = z_wsi.index_select(dim=0, index=rna_indices)
            z_rna_any: Any = self._model.forward_rna(
                gene_ids=gene_ids.index_select(dim=0, index=rna_indices),
                expr_vals=expr_vals.index_select(dim=0, index=rna_indices),
                gene_mask=gene_mask.index_select(dim=0, index=rna_indices),
            )
            z_rna: torch.Tensor = self._coerce_tensor(z_rna_any, name="z_rna")
            self._validate_embedding_tensor(z_rna, name="z_rna")

            loss_rna: torch.Tensor = self._compute_contrastive(z_wsi_rna, z_rna)
            self._validate_loss_tensor(loss_rna, name="loss_rna")

            loss_terms.append(loss_rna)
            loss_weights.append(float(self._weight_rna))
            metrics["train/loss_wsi_rna"] = float(loss_rna.detach().cpu().item())
            metrics["train/n_pairs_rna"] = float(int(rna_indices.numel()))

        # DNA branch.
        dna_indices: torch.Tensor = self._resolve_modality_indices(
            batch=batch,
            modality_flag_key=BATCH_KEY_HAS_DNA,
            batch_size=batch_size,
        )
        if int(dna_indices.numel()) > 0:
            dna_multi_hot: torch.Tensor = self._coerce_tensor(
                self._require_key(batch, BATCH_KEY_DNA_MULTI_HOT),
                name=BATCH_KEY_DNA_MULTI_HOT,
            )
            self._validate_first_dim(dna_multi_hot, batch_size, BATCH_KEY_DNA_MULTI_HOT)
            self._validate_dna_shape(dna_multi_hot)

            z_wsi_dna: torch.Tensor = z_wsi.index_select(dim=0, index=dna_indices)
            z_dna_any: Any = self._model.forward_dna(
                dna_multi_hot=dna_multi_hot.index_select(dim=0, index=dna_indices)
            )
            z_dna: torch.Tensor = self._coerce_tensor(z_dna_any, name="z_dna")
            self._validate_embedding_tensor(z_dna, name="z_dna")

            loss_dna: torch.Tensor = self._compute_contrastive(z_wsi_dna, z_dna)
            self._validate_loss_tensor(loss_dna, name="loss_dna")

            loss_terms.append(loss_dna)
            loss_weights.append(float(self._weight_dna))
            metrics["train/loss_wsi_dna"] = float(loss_dna.detach().cpu().item())
            metrics["train/n_pairs_dna"] = float(int(dna_indices.numel()))

        if len(loss_terms) == 0:
            if self._fail_if_no_valid_pair:
                raise PretrainModuleInputError(
                    "No valid modality pairs in batch; cannot compute pretraining loss."
                )
            # Provide zero-loss placeholder on same device for non-failing mode.
            zero_loss: torch.Tensor = torch.zeros((), dtype=z_wsi.dtype, device=z_wsi.device)
            metrics["train/loss_total"] = 0.0
            return zero_loss, self._prefix_metrics(metrics, train_mode=train_mode), z_wsi.detach()

        total_loss: torch.Tensor = self._weighted_mean(loss_terms, loss_weights)
        self._validate_loss_tensor(total_loss, name="loss_total")

        metrics["train/loss_total"] = float(total_loss.detach().cpu().item())

        return total_loss, self._prefix_metrics(metrics, train_mode=train_mode), z_wsi.detach()

    def _compute_contrastive(self, z_wsi: torch.Tensor, z_mol: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss using model-attached loss_fn when available."""
        if hasattr(self._model, "_loss_fn"):
            loss_module: Any = getattr(self._model, "_loss_fn")
            if callable(loss_module):
                output: Any = loss_module(z_wsi=z_wsi, z_mol=z_mol)
                tensor_output: torch.Tensor = self._coerce_tensor(output, name="contrastive_loss")
                if tensor_output.ndim > 0:
                    tensor_output = tensor_output.mean()
                return tensor_output

        # Fallback not expected in this project layout.
        raise PretrainModuleRuntimeError(
            "ThreadsModel does not expose a usable contrastive loss function."
        )

    def _weighted_mean(self, loss_terms: Sequence[torch.Tensor], weights: Sequence[float]) -> torch.Tensor:
        if len(loss_terms) == 0:
            raise PretrainModuleRuntimeError("Cannot aggregate empty loss list.")
        if len(loss_terms) != len(weights):
            raise PretrainModuleRuntimeError("Loss term count and weight count mismatch.")

        base_device: torch.device = loss_terms[0].device
        weight_tensor: torch.Tensor = torch.as_tensor(weights, dtype=torch.float32, device=base_device)
        if (weight_tensor <= 0.0).any():
            raise PretrainModuleRuntimeError(
                f"All modality weights must be > 0, got {weight_tensor.tolist()}."
            )

        stacked_loss: torch.Tensor = torch.stack([item.to(torch.float32) for item in loss_terms], dim=0)
        weighted_sum: torch.Tensor = torch.sum(stacked_loss * weight_tensor)
        total_weight: torch.Tensor = weight_tensor.sum().clamp_min(DEFAULT_METRIC_EPS)
        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # Optimizer/scheduler resolution
    # ------------------------------------------------------------------
    def _resolve_optimizer_config(self, raw_cfg: Mapping[str, Any]) -> _OptimizerConfig:
        normalized_name: str = self._as_str(
            self._first_present(raw_cfg, (("name",), ("optimizer",),)),
            key="optimizer.name",
            default=DEFAULT_OPTIMIZER_NAME,
        )
        if normalized_name.strip().lower() != DEFAULT_OPTIMIZER_NAME.lower():
            raise PretrainModuleConfigError(
                f"Optimizer must be {DEFAULT_OPTIMIZER_NAME}, got {normalized_name!r}."
            )

        learning_rate: float = self._as_float(
            self._first_present(raw_cfg, (("learning_rate",), ("lr",))),
            key="optimizer.learning_rate",
            default=DEFAULT_LEARNING_RATE,
        )
        if not math.isclose(learning_rate, DEFAULT_LEARNING_RATE, rel_tol=0.0, abs_tol=1e-16):
            raise PretrainModuleConfigError(
                "learning_rate must match paper/config invariant 1e-5; "
                f"got {learning_rate}."
            )

        betas_value: Any = self._first_present(raw_cfg, (("betas",),), None)
        if betas_value is None:
            raise PretrainModuleConfigError(
                "optimizer.betas is required for pretrain stage and cannot be null."
            )
        betas: Tuple[float, float] = self._parse_betas(betas_value)

        eps: float = self._as_float(
            self._first_present(raw_cfg, (("eps",),), DEFAULT_EPS),
            key="optimizer.eps",
            default=DEFAULT_EPS,
        )

        weight_decay_value: Any = self._first_present(raw_cfg, (("weight_decay",),), None)
        if weight_decay_value is None:
            raise PretrainModuleConfigError(
                "optimizer.weight_decay is required for pretrain stage and cannot be null."
            )
        weight_decay: float = self._as_float(
            weight_decay_value,
            key="optimizer.weight_decay",
            default=DEFAULT_WEIGHT_DECAY,
        )

        zero_grad_set_to_none: bool = self._as_bool(
            self._first_present(raw_cfg, (("zero_grad_set_to_none",),), True),
            key="optimizer.zero_grad_set_to_none",
            default=True,
        )

        return _OptimizerConfig(
            name=DEFAULT_OPTIMIZER_NAME,
            learning_rate=float(learning_rate),
            betas=(float(betas[0]), float(betas[1])),
            eps=float(eps),
            weight_decay=float(weight_decay),
            zero_grad_set_to_none=bool(zero_grad_set_to_none),
        )

    def _resolve_scheduler_config(
        self,
        raw_cfg: Mapping[str, Any],
        *,
        optimizer_lr: float,
    ) -> _SchedulerConfig:
        scheduler_type: str = self._as_str(
            self._first_present(raw_cfg, (("type",), ("scheduler",))),
            key="scheduler.type",
            default=DEFAULT_SCHEDULER_TYPE,
        )
        if scheduler_type.strip().lower() != DEFAULT_SCHEDULER_TYPE.lower():
            raise PretrainModuleConfigError(
                f"scheduler.type must be '{DEFAULT_SCHEDULER_TYPE}', got {scheduler_type!r}."
            )

        max_epochs: int = self._as_int(
            self._first_present(raw_cfg, (("max_epochs",),)),
            key="scheduler.max_epochs",
            default=DEFAULT_MAX_EPOCHS,
        )
        if int(max_epochs) != DEFAULT_MAX_EPOCHS:
            raise PretrainModuleConfigError(
                f"scheduler.max_epochs must be {DEFAULT_MAX_EPOCHS}, got {max_epochs}."
            )

        warmup_cfg: Mapping[str, Any] = self._to_dict(
            self._first_present(raw_cfg, (("warmup",),), {})
        )
        warmup_enabled: bool = self._as_bool(
            self._first_present(warmup_cfg, (("enabled",),), True),
            key="scheduler.warmup.enabled",
            default=True,
        )
        warmup_epochs: int = self._as_int(
            self._first_present(warmup_cfg, (("epochs",),), DEFAULT_WARMUP_EPOCHS),
            key="scheduler.warmup.epochs",
            default=DEFAULT_WARMUP_EPOCHS,
        )
        if int(warmup_epochs) != DEFAULT_WARMUP_EPOCHS:
            raise PretrainModuleConfigError(
                f"scheduler.warmup.epochs must be {DEFAULT_WARMUP_EPOCHS}, got {warmup_epochs}."
            )

        warmup_start_lr: float = self._as_float(
            self._first_present(warmup_cfg, (("start_lr",),), DEFAULT_WARMUP_START_LR),
            key="scheduler.warmup.start_lr",
            default=DEFAULT_WARMUP_START_LR,
        )
        peak_lr: float = self._as_float(
            self._first_present(warmup_cfg, (("peak_lr",),), DEFAULT_PEAK_LR),
            key="scheduler.warmup.peak_lr",
            default=DEFAULT_PEAK_LR,
        )
        if not math.isclose(peak_lr, optimizer_lr, rel_tol=0.0, abs_tol=1e-16):
            raise PretrainModuleConfigError(
                "scheduler warmup peak_lr must match optimizer learning_rate; "
                f"peak_lr={peak_lr}, optimizer_lr={optimizer_lr}."
            )

        cosine_cfg: Mapping[str, Any] = self._to_dict(
            self._first_present(raw_cfg, (("cosine",),), {})
        )
        final_lr_value: Any = self._first_present(cosine_cfg, (("final_lr",),), None)
        if final_lr_value is None:
            raise PretrainModuleConfigError(
                "scheduler.cosine.final_lr is required for pretrain stage and cannot be null."
            )
        final_lr: float = self._as_float(
            final_lr_value,
            key="scheduler.cosine.final_lr",
            default=DEFAULT_FINAL_LR,
        )

        return _SchedulerConfig(
            scheduler_type=DEFAULT_SCHEDULER_TYPE,
            max_epochs=int(max_epochs),
            warmup_enabled=bool(warmup_enabled),
            warmup_epochs=int(warmup_epochs),
            warmup_start_lr=float(warmup_start_lr),
            peak_lr=float(peak_lr),
            final_lr=float(final_lr),
        )

    def _build_lr_lambda(self) -> Any:
        warmup_epochs: int = int(self._scheduler_config.warmup_epochs)
        max_epochs: int = int(self._scheduler_config.max_epochs)

        peak_lr: float = float(self._scheduler_config.peak_lr)
        start_lr: float = float(self._scheduler_config.warmup_start_lr)
        final_lr: float = float(self._scheduler_config.final_lr)

        base_lr: float = float(self._optimizer_config.learning_rate)
        if base_lr <= 0.0:
            raise PretrainModuleConfigError(f"optimizer.learning_rate must be > 0, got {base_lr}.")

        start_ratio: float = start_lr / base_lr
        final_ratio: float = final_lr / base_lr

        # Guard against invalid scheduler horizon.
        cosine_total_epochs: int = max(1, max_epochs - warmup_epochs)

        def _lr_lambda(epoch: int) -> float:
            epoch_index: int = int(epoch)

            if self._scheduler_config.warmup_enabled and epoch_index < warmup_epochs:
                # Linear warmup from start_ratio to 1.0 over warmup_epochs.
                progress: float = float(epoch_index + 1) / float(max(1, warmup_epochs))
                return float(start_ratio + (1.0 - start_ratio) * progress)

            # Cosine decay from 1.0 to final_ratio.
            post_warmup_epoch: int = max(0, epoch_index - warmup_epochs)
            cosine_progress: float = min(1.0, float(post_warmup_epoch + 1) / float(cosine_total_epochs))
            cosine_value: float = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            return float(final_ratio + (1.0 - final_ratio) * cosine_value)

        return _lr_lambda

    # ------------------------------------------------------------------
    # Logging and diagnostics
    # ------------------------------------------------------------------
    def _log_train_step_metrics(self, metrics: Mapping[str, float]) -> None:
        batch_size_value: int = int(self._infer_batch_size_from_metrics(metrics))

        for metric_name, metric_value in metrics.items():
            if not metric_name.startswith("train/"):
                continue
            self.log(
                metric_name,
                float(metric_value),
                on_step=True,
                on_epoch=False,
                prog_bar=(metric_name == "train/loss_total"),
                logger=True,
                sync_dist=self._log_sync_dist,
                batch_size=batch_size_value,
            )

        learning_rate: float = float(self._current_learning_rate())
        self.log(
            "train/lr",
            learning_rate,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )

        # Expose warmup/rank monitoring metadata.
        current_epoch_index: int = int(getattr(self, "current_epoch", 0))
        rankme_active: float = 1.0 if current_epoch_index >= int(self.rankme_start_epoch) else 0.0
        self.log(
            "train/rankme_monitor_active",
            rankme_active,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=self._log_sync_dist,
            batch_size=batch_size_value,
        )

    def _safe_model_diagnostics(self, batch: Mapping[str, Any]) -> Dict[str, float]:
        """Best-effort float diagnostics from ThreadsModel.training_step."""
        try:
            payload: Any = self._model.training_step(dict(batch))
        except Exception:
            return {}

        if not isinstance(payload, Mapping):
            return {}

        output: Dict[str, float] = {}
        for key, value in payload.items():
            metric_name: str = str(key)
            if not metric_name:
                continue
            numeric_value: Optional[float] = self._safe_float(value)
            if numeric_value is None:
                continue
            if metric_name.startswith("loss_"):
                output[f"train/{metric_name}"] = numeric_value
            elif metric_name in {"loss", "loss_total", "n_pairs_rna", "n_pairs_dna"}:
                output[f"train/{metric_name}"] = numeric_value
        return output

    def _prefix_metrics(self, metrics: Mapping[str, float], train_mode: bool) -> Dict[str, float]:
        prefix: str = "train/" if train_mode else "val/"
        output: Dict[str, float] = {}
        for key, value in metrics.items():
            normalized_key: str = str(key)
            if normalized_key.startswith("train/") or normalized_key.startswith("val/"):
                if train_mode:
                    output[normalized_key] = float(value)
                else:
                    tail: str = normalized_key.split("/", 1)[-1]
                    output[f"{prefix}{tail}"] = float(value)
            else:
                output[f"{prefix}{normalized_key}"] = float(value)
        return output

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate_model(self, model: ThreadsModel) -> ThreadsModel:
        if not isinstance(model, ThreadsModel):
            raise PretrainModuleConfigError(
                f"model must be ThreadsModel, got {type(model).__name__}."
            )
        return model

    def _validate_embedding_tensor(self, tensor_value: torch.Tensor, name: str) -> None:
        if tensor_value.ndim != 2:
            raise PretrainModuleInputError(
                f"{name} must be rank-2 [B,D], got {tuple(tensor_value.shape)}."
            )
        if int(tensor_value.shape[1]) != DEFAULT_EMBEDDING_DIM:
            raise PretrainModuleInputError(
                f"{name} embedding dim mismatch: expected {DEFAULT_EMBEDDING_DIM}, got {int(tensor_value.shape[1])}."
            )
        if self._validate_numerics and not torch.isfinite(tensor_value).all():
            raise PretrainModuleInputError(f"{name} contains NaN/Inf.")

    def _validate_loss_tensor(self, tensor_value: torch.Tensor, name: str) -> None:
        if not isinstance(tensor_value, torch.Tensor):
            raise PretrainModuleRuntimeError(f"{name} must be torch.Tensor.")
        if tensor_value.ndim > 1:
            raise PretrainModuleRuntimeError(
                f"{name} must be scalar-like tensor, got shape={tuple(tensor_value.shape)}."
            )
        if self._validate_numerics and not torch.isfinite(tensor_value):
            raise PretrainModuleRuntimeError(f"{name} is non-finite.")

    def _validate_first_dim(self, tensor_value: torch.Tensor, expected: int, name: str) -> None:
        if tensor_value.ndim <= 0:
            raise PretrainModuleInputError(f"{name} must contain batch dimension.")
        if int(tensor_value.shape[0]) != int(expected):
            raise PretrainModuleInputError(
                f"{name} batch mismatch: expected {expected}, got {int(tensor_value.shape[0])}."
            )

    def _validate_dna_shape(self, dna_tensor: torch.Tensor) -> None:
        if dna_tensor.ndim == 1:
            dna_tensor = dna_tensor.unsqueeze(0)
        if dna_tensor.ndim != 2:
            raise PretrainModuleInputError(
                f"{BATCH_KEY_DNA_MULTI_HOT} must be rank-2 [B,{DEFAULT_DNA_INPUT_DIM}], got {tuple(dna_tensor.shape)}."
            )
        if int(dna_tensor.shape[1]) != DEFAULT_DNA_INPUT_DIM:
            raise PretrainModuleInputError(
                f"{BATCH_KEY_DNA_MULTI_HOT} width mismatch: expected {DEFAULT_DNA_INPUT_DIM}, got {int(dna_tensor.shape[1])}."
            )

    def _resolve_modality_indices(
        self,
        *,
        batch: Mapping[str, Any],
        modality_flag_key: str,
        batch_size: int,
    ) -> torch.Tensor:
        if modality_flag_key not in batch:
            return torch.arange(batch_size, dtype=torch.long, device=self.device)

        flag_tensor: torch.Tensor = self._coerce_tensor(batch[modality_flag_key], name=modality_flag_key)
        if flag_tensor.ndim != 1:
            raise PretrainModuleInputError(
                f"{modality_flag_key} must be rank-1 [B], got {tuple(flag_tensor.shape)}."
            )
        if int(flag_tensor.shape[0]) != int(batch_size):
            raise PretrainModuleInputError(
                f"{modality_flag_key} length mismatch: expected {batch_size}, got {int(flag_tensor.shape[0])}."
            )

        if flag_tensor.dtype != torch.bool:
            if torch.is_floating_point(flag_tensor):
                flag_tensor = flag_tensor > 0.5
            else:
                flag_tensor = flag_tensor > 0

        return torch.nonzero(flag_tensor, as_tuple=False).flatten().to(torch.long)

    def _require_key(self, mapping: Mapping[str, Any], key: str) -> Any:
        if key not in mapping:
            raise PretrainModuleInputError(f"Missing required batch key: {key!r}")
        return mapping[key]

    def _coerce_tensor(self, value: Any, name: str) -> torch.Tensor:
        try:
            tensor_value: torch.Tensor = torch.as_tensor(value)
        except Exception as exc:  # noqa: BLE001
            raise PretrainModuleInputError(
                f"{name} cannot be converted to tensor: {exc}"
            ) from exc

        if torch.is_floating_point(tensor_value):
            tensor_value = tensor_value.to(torch.float32)
            if self._validate_numerics and not torch.isfinite(tensor_value).all():
                raise PretrainModuleInputError(f"{name} contains NaN/Inf values.")

        return tensor_value

    def _current_learning_rate(self) -> float:
        optimizer: Optional[torch.optim.Optimizer] = self.optimizers() if self.trainer is not None else None
        if optimizer is None:
            return float(self._optimizer_config.learning_rate)
        if len(optimizer.param_groups) == 0:
            return float(self._optimizer_config.learning_rate)
        return float(optimizer.param_groups[0].get("lr", self._optimizer_config.learning_rate))

    def _infer_batch_size(self, batch: Mapping[str, Any]) -> int:
        if BATCH_KEY_PATCH_FEATURES not in batch:
            return 1
        patch_tensor: torch.Tensor = self._coerce_tensor(batch[BATCH_KEY_PATCH_FEATURES], name=BATCH_KEY_PATCH_FEATURES)
        if patch_tensor.ndim < 1:
            return 1
        return int(patch_tensor.shape[0]) if patch_tensor.ndim >= 3 else 1

    def _infer_batch_size_from_metrics(self, _metrics: Mapping[str, float]) -> int:
        # Step-level logging currently does not carry batch payload here.
        return 1

    # ------------------------------------------------------------------
    # Config parsing helpers
    # ------------------------------------------------------------------
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
        raise PretrainModuleConfigError(
            f"Unsupported config object type: {type(value).__name__}."
        )

    @staticmethod
    def _deep_get(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
        cursor: Any = mapping
        for key in path:
            if not isinstance(cursor, Mapping):
                return default
            if key not in cursor:
                return default
            cursor = cursor[key]
        return cursor

    @classmethod
    def _first_present(
        cls,
        mapping: Mapping[str, Any],
        paths: Sequence[Sequence[str]],
        default: Any = None,
    ) -> Any:
        for path in paths:
            value: Any = cls._deep_get(mapping, path, None)
            if value is not None:
                return value
        return default

    def _resolve_modality_weight(self, cfg: Mapping[str, Any], modality: str, default: float) -> float:
        raw_value: Any = self._first_present(
            cfg,
            paths=(("loss_weights", modality),),
            default=default,
        )
        value: float = self._as_float(raw_value, key=f"loss_weights.{modality}", default=default)
        if value <= 0.0:
            raise PretrainModuleConfigError(
                f"loss weight for {modality} must be > 0, got {value}."
            )
        return value

    def _resolve_bool(
        self,
        cfg: Mapping[str, Any],
        *,
        paths: Sequence[Sequence[str]],
        default: bool,
    ) -> bool:
        raw_value: Any = self._first_present(cfg, paths=paths, default=default)
        return self._as_bool(raw_value, key="bool_flag", default=default)

    @staticmethod
    def _parse_betas(value: Any) -> Tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            beta_1: float = float(value[0])
            beta_2: float = float(value[1])
            if not (0.0 < beta_1 < 1.0 and 0.0 < beta_2 < 1.0):
                raise PretrainModuleConfigError(
                    f"optimizer.betas must be in (0,1), got {(beta_1, beta_2)}"
                )
            return beta_1, beta_2
        raise PretrainModuleConfigError(
            f"optimizer.betas must be length-2 sequence, got {value!r}."
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
            raise PretrainModuleConfigError(f"{key} must be int, got bool.")
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if not value.is_integer():
                raise PretrainModuleConfigError(f"{key} must be integer-like, got {value}.")
            return int(value)
        if isinstance(value, str):
            stripped: str = value.strip()
            if stripped == "":
                return int(default)
            try:
                return int(stripped)
            except ValueError as exc:
                raise PretrainModuleConfigError(f"{key} must be int, got {value!r}.") from exc
        raise PretrainModuleConfigError(f"{key} must be int, got {type(value).__name__}.")

    @staticmethod
    def _as_float(value: Any, *, key: str, default: float) -> float:
        if value is None:
            return float(default)
        if isinstance(value, bool):
            raise PretrainModuleConfigError(f"{key} must be float, got bool.")
        if isinstance(value, (int, float)):
            parsed: float = float(value)
            if not math.isfinite(parsed):
                raise PretrainModuleConfigError(f"{key} must be finite, got {parsed}.")
            return parsed
        if isinstance(value, str):
            stripped: str = value.strip()
            if stripped == "":
                return float(default)
            try:
                parsed = float(stripped)
            except ValueError as exc:
                raise PretrainModuleConfigError(f"{key} must be float, got {value!r}.") from exc
            if not math.isfinite(parsed):
                raise PretrainModuleConfigError(f"{key} must be finite, got {parsed}.")
            return parsed
        raise PretrainModuleConfigError(f"{key} must be float, got {type(value).__name__}.")

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
        raise PretrainModuleConfigError(f"{key} must be bool, got {value!r}.")

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            parsed: float = float(value)
        except Exception:
            return None
        if not math.isfinite(parsed):
            return None
        return parsed


__all__ = [
    "PretrainModuleError",
    "PretrainModuleConfigError",
    "PretrainModuleInputError",
    "PretrainModuleRuntimeError",
    "PretrainModule",
]
