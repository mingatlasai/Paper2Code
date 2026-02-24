"""Loss functions for TITAN stage-1/2/3 training.

This module implements:
- `IBOTLoss` for stage-1 feature-space iBOT distillation/masked modeling.
- `contrastive_info_nce_loss` for stage-2/3 image-text alignment.
- `caption_cross_entropy_loss` for stage-2/3 caption/report generation.
- `compute_multimodal_stage_loss` for weighted stage-2/3 composition.

The implementation follows provided config constraints:
- patch size: 512
- magnification: 20x
- feature dim / embedding dim: 768
- stage-1 context: 16x16 grid with 2 global + 10 local crops
- stage-3 context: 64x64 grid

Notes on unresolved supplementary hyperparameters:
- `student_temp`, `teacher_temp`, `center_momentum`, and stage-2/3 loss
  weights are configurable and may be unknown from the provided paper excerpt.
- Defaults are explicit, safe fallbacks and should be overridden via config when
  exact values are available.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE1_GLOBAL_VIEWS: int = 2
_STAGE1_GLOBAL_GRID: Tuple[int, int] = (14, 14)
_STAGE1_LOCAL_VIEWS: int = 10
_STAGE1_LOCAL_GRID: Tuple[int, int] = (6, 6)

_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

# Explicit fallbacks for paper-missing supplementary hyperparameters.
_DEFAULT_STUDENT_TEMP: float = 1.0
_DEFAULT_TEACHER_TEMP: float = 1.0
_DEFAULT_CENTER_MOMENTUM: float = 0.9
_DEFAULT_CONTRASTIVE_WEIGHT: float = 1.0
_DEFAULT_CAPTION_WEIGHT: float = 1.0

_DEFAULT_IGNORE_INDEX: int = -100
_DEFAULT_EPS: float = 1.0e-12


class LossesError(RuntimeError):
    """Base exception for loss-related failures."""


class LossShapeError(LossesError):
    """Raised when incoming tensors violate expected shape contracts."""


class IBOTLoss(nn.Module):
    """iBOT loss for feature-space stage-1 TITAN pretraining.

    Public API is design-locked:
    - `__init__(student_temp, teacher_temp, center_momentum)`
    - `compute(student_out, teacher_out, mask)`
    - `update_center(teacher_out)`

    The implementation supports:
    - token tensors shaped `[B, T, D]`, `[T, D]`, `[B, H, W, D]`, or flattened
      to 2D with matching mask.
    - mask semantics: `True/1` means "masked token".
    """

    def __init__(
        self,
        student_temp: float = _DEFAULT_STUDENT_TEMP,
        teacher_temp: float = _DEFAULT_TEACHER_TEMP,
        center_momentum: float = _DEFAULT_CENTER_MOMENTUM,
    ) -> None:
        """Initialize iBOT loss state.

        Args:
            student_temp: Student temperature (>0).
            teacher_temp: Teacher temperature (>0).
            center_momentum: EMA momentum for teacher center in [0, 1).
        """
        super().__init__()

        if not isinstance(student_temp, (int, float)):
            raise TypeError("student_temp must be numeric.")
        if not isinstance(teacher_temp, (int, float)):
            raise TypeError("teacher_temp must be numeric.")
        if not isinstance(center_momentum, (int, float)):
            raise TypeError("center_momentum must be numeric.")

        student_temp_value: float = float(student_temp)
        teacher_temp_value: float = float(teacher_temp)
        center_momentum_value: float = float(center_momentum)

        if student_temp_value <= 0.0:
            raise ValueError("student_temp must be > 0.")
        if teacher_temp_value <= 0.0:
            raise ValueError("teacher_temp must be > 0.")
        if center_momentum_value < 0.0 or center_momentum_value >= 1.0:
            raise ValueError("center_momentum must be in [0, 1).")

        self.student_temp: float = student_temp_value
        self.teacher_temp: float = teacher_temp_value
        self.center_momentum: float = center_momentum_value

        # Registered lazily on first valid call because projection dim can vary.
        self.register_buffer("center", torch.zeros(1, 1, dtype=torch.float32), persistent=True)
        self._center_initialized: bool = False

        # Debug/stat hooks for trainer logging.
        self.last_stats: Dict[str, float] = {}

        # Provenance constants for reproducibility logging.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage1_global_views: int = _STAGE1_GLOBAL_VIEWS
        self.stage1_global_grid: Tuple[int, int] = _STAGE1_GLOBAL_GRID
        self.stage1_local_views: int = _STAGE1_LOCAL_VIEWS
        self.stage1_local_grid: Tuple[int, int] = _STAGE1_LOCAL_GRID

    def compute(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute iBOT distillation + masked-token loss.

        Args:
            student_out: Student projections aligned with teacher tokens.
            teacher_out: Teacher projections aligned with student tokens.
            mask: Token mask (`True/1` means masked token).

        Returns:
            Scalar tensor loss.
        """
        student_btd, teacher_btd, token_mask_bt = self._normalize_inputs(
            student_out=student_out,
            teacher_out=teacher_out,
            mask=mask,
        )

        self._ensure_center_dim(feature_dim=int(teacher_btd.shape[-1]), device=teacher_btd.device)

        # Teacher targets: centered, temperature-scaled, detached probabilities.
        teacher_logits: torch.Tensor = (teacher_btd - self.center.view(1, 1, -1)) / self.teacher_temp
        teacher_prob: torch.Tensor = F.softmax(teacher_logits, dim=-1).detach()

        # Student log-probabilities.
        student_logits: torch.Tensor = student_btd / self.student_temp
        student_log_prob: torch.Tensor = F.log_softmax(student_logits, dim=-1)

        # Token-level cross entropy: shape [B, T].
        token_ce: torch.Tensor = -(teacher_prob * student_log_prob).sum(dim=-1)

        # Distillation over all valid tokens.
        valid_float: torch.Tensor = torch.ones_like(token_ce, dtype=token_ce.dtype)
        distill_den: torch.Tensor = valid_float.sum().clamp_min(1.0)
        loss_distill: torch.Tensor = (token_ce * valid_float).sum() / distill_den

        # Masked-token term only where mask == True.
        mask_float: torch.Tensor = token_mask_bt.to(dtype=token_ce.dtype)
        masked_count: torch.Tensor = mask_float.sum()
        if float(masked_count.item()) > 0.0:
            loss_masked: torch.Tensor = (token_ce * mask_float).sum() / masked_count.clamp_min(1.0)
            total_loss: torch.Tensor = 0.5 * (loss_distill + loss_masked)
        else:
            loss_masked = torch.zeros((), device=token_ce.device, dtype=token_ce.dtype)
            total_loss = loss_distill

        self.update_center(teacher_out=teacher_btd)

        self.last_stats = {
            "loss_distill": float(loss_distill.detach().cpu().item()),
            "loss_masked": float(loss_masked.detach().cpu().item()),
            "loss_total": float(total_loss.detach().cpu().item()),
            "masked_tokens": float(masked_count.detach().cpu().item()),
        }

        if not torch.isfinite(total_loss):
            raise LossesError("IBOTLoss produced non-finite loss value.")

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_out: torch.Tensor) -> None:
        """Update running teacher center via EMA.

        Args:
            teacher_out: Teacher projections with shape `[B,T,D]` or compatible.
        """
        if not isinstance(teacher_out, torch.Tensor):
            raise TypeError(f"teacher_out must be torch.Tensor, got {type(teacher_out).__name__}.")

        teacher_btd: torch.Tensor = _to_btd(teacher_out)
        feature_dim: int = int(teacher_btd.shape[-1])
        self._ensure_center_dim(feature_dim=feature_dim, device=teacher_btd.device)

        batch_center: torch.Tensor = teacher_btd.detach().mean(dim=(0, 1))

        if _dist_is_active():
            # Global mean across ranks for stable center in DDP.
            world_size: int = dist.get_world_size()
            dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
            batch_center = batch_center / float(world_size)

        self.center.mul_(self.center_momentum).add_(batch_center * (1.0 - self.center_momentum))

    def _ensure_center_dim(self, feature_dim: int, device: torch.device) -> None:
        if feature_dim <= 0:
            raise LossShapeError("feature_dim must be > 0 for center initialization.")

        if (not self._center_initialized) or int(self.center.numel()) != feature_dim:
            center_init: torch.Tensor = torch.zeros(feature_dim, device=device, dtype=torch.float32)
            self.center = center_init
            self._center_initialized = True
        elif self.center.device != device:
            self.center = self.center.to(device=device)

    def _normalize_inputs(
        self,
        student_out: torch.Tensor,
        teacher_out: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(student_out, torch.Tensor):
            raise TypeError(f"student_out must be torch.Tensor, got {type(student_out).__name__}.")
        if not isinstance(teacher_out, torch.Tensor):
            raise TypeError(f"teacher_out must be torch.Tensor, got {type(teacher_out).__name__}.")
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"mask must be torch.Tensor, got {type(mask).__name__}.")

        student_btd: torch.Tensor = _to_btd(student_out)
        teacher_btd: torch.Tensor = _to_btd(teacher_out)

        if tuple(student_btd.shape) != tuple(teacher_btd.shape):
            raise LossShapeError(
                "student_out and teacher_out must align after normalization. "
                f"Got {tuple(student_btd.shape)} vs {tuple(teacher_btd.shape)}."
            )

        batch_size: int = int(student_btd.shape[0])
        num_tokens: int = int(student_btd.shape[1])

        mask_bt: torch.Tensor = _to_bt_mask(mask, batch_size=batch_size, num_tokens=num_tokens)

        return (
            student_btd.to(dtype=torch.float32),
            teacher_btd.to(dtype=torch.float32),
            mask_bt.to(dtype=torch.bool),
        )


def contrastive_info_nce_loss(
    image_embeddings: Optional[torch.Tensor] = None,
    text_embeddings: Optional[torch.Tensor] = None,
    logits_per_image: Optional[torch.Tensor] = None,
    logits_per_text: Optional[torch.Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    normalize_embeddings: bool = True,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Compute symmetric image-text InfoNCE loss.

    You can pass either:
    - normalized/un-normalized embeddings (`image_embeddings`, `text_embeddings`), or
    - precomputed logits (`logits_per_image`, optionally `logits_per_text`).

    Args:
        image_embeddings: Tensor [B, D].
        text_embeddings: Tensor [B, D].
        logits_per_image: Tensor [B, B].
        logits_per_text: Optional tensor [B, B]. If omitted, transpose of image logits.
        logit_scale: Optional scalar or broadcastable tensor multiplier.
        normalize_embeddings: If True, L2-normalize embeddings before logits.
        reduction: Cross-entropy reduction, usually "mean".

    Returns:
        Dict with keys:
        - `loss_contrastive`
        - `loss_i2t`
        - `loss_t2i`
        - `acc_i2t`
        - `acc_t2i`
        - `logits_per_image`
        - `logits_per_text`
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction='{reduction}'.")

    if logits_per_image is None:
        if image_embeddings is None or text_embeddings is None:
            raise ValueError(
                "Either provide logits_per_image or both image_embeddings and text_embeddings."
            )

        if not isinstance(image_embeddings, torch.Tensor):
            raise TypeError("image_embeddings must be torch.Tensor.")
        if not isinstance(text_embeddings, torch.Tensor):
            raise TypeError("text_embeddings must be torch.Tensor.")
        if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
            raise LossShapeError(
                "image_embeddings and text_embeddings must have shape [B,D]. "
                f"Got {tuple(image_embeddings.shape)} and {tuple(text_embeddings.shape)}."
            )
        if int(image_embeddings.shape[0]) != int(text_embeddings.shape[0]):
            raise LossShapeError(
                "Embedding batch sizes must match. "
                f"Got image B={int(image_embeddings.shape[0])}, text B={int(text_embeddings.shape[0])}."
            )
        if int(image_embeddings.shape[1]) != int(text_embeddings.shape[1]):
            raise LossShapeError(
                "Embedding dimensions must match. "
                f"Got image D={int(image_embeddings.shape[1])}, text D={int(text_embeddings.shape[1])}."
            )

        image_emb: torch.Tensor = image_embeddings.to(dtype=torch.float32)
        text_emb: torch.Tensor = text_embeddings.to(dtype=torch.float32)

        if normalize_embeddings:
            image_emb = F.normalize(image_emb, p=2.0, dim=-1, eps=_DEFAULT_EPS)
            text_emb = F.normalize(text_emb, p=2.0, dim=-1, eps=_DEFAULT_EPS)

        logits_per_image = torch.matmul(image_emb, text_emb.transpose(0, 1))

        if logit_scale is not None:
            if not isinstance(logit_scale, torch.Tensor):
                logit_scale_t: torch.Tensor = torch.tensor(float(logit_scale), device=logits_per_image.device)
            else:
                logit_scale_t = logit_scale.to(device=logits_per_image.device, dtype=logits_per_image.dtype)
            logits_per_image = logits_per_image * logit_scale_t

    if not isinstance(logits_per_image, torch.Tensor):
        raise TypeError("logits_per_image must be torch.Tensor.")
    if logits_per_image.ndim != 2:
        raise LossShapeError(
            f"logits_per_image must have shape [B,B], got {tuple(logits_per_image.shape)}."
        )

    batch_size: int = int(logits_per_image.shape[0])
    if batch_size <= 0 or int(logits_per_image.shape[1]) != batch_size:
        raise LossShapeError(
            "logits_per_image must be square [B,B]. "
            f"Got {tuple(logits_per_image.shape)}."
        )

    if logits_per_text is None:
        logits_per_text = logits_per_image.transpose(0, 1)
    if not isinstance(logits_per_text, torch.Tensor):
        raise TypeError("logits_per_text must be torch.Tensor.")
    if tuple(logits_per_text.shape) != (batch_size, batch_size):
        raise LossShapeError(
            "logits_per_text must match shape [B,B]. "
            f"Got {tuple(logits_per_text.shape)} expected {(batch_size, batch_size)}."
        )

    targets: torch.Tensor = torch.arange(batch_size, device=logits_per_image.device, dtype=torch.long)

    loss_i2t: torch.Tensor = F.cross_entropy(logits_per_image, targets, reduction=reduction)
    loss_t2i: torch.Tensor = F.cross_entropy(logits_per_text, targets, reduction=reduction)

    if reduction == "none":
        loss_contrastive: torch.Tensor = 0.5 * (loss_i2t + loss_t2i)
    else:
        loss_contrastive = 0.5 * (loss_i2t + loss_t2i)

    with torch.no_grad():
        pred_i2t: torch.Tensor = torch.argmax(logits_per_image, dim=-1)
        pred_t2i: torch.Tensor = torch.argmax(logits_per_text, dim=-1)
        acc_i2t: torch.Tensor = (pred_i2t == targets).to(dtype=torch.float32).mean()
        acc_t2i: torch.Tensor = (pred_t2i == targets).to(dtype=torch.float32).mean()

    return {
        "loss_contrastive": loss_contrastive,
        "loss_i2t": loss_i2t,
        "loss_t2i": loss_t2i,
        "acc_i2t": acc_i2t,
        "acc_t2i": acc_t2i,
        "logits_per_image": logits_per_image,
        "logits_per_text": logits_per_text,
    }


def caption_cross_entropy_loss(
    decoder_logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
    reduction: str = "mean",
) -> Dict[str, torch.Tensor]:
    """Compute token-level caption/report generation cross-entropy.

    Args:
        decoder_logits: Tensor [B, L, V].
        labels: Tensor [B, L].
        ignore_index: Ignore index for padded/invalid labels.
        reduction: Reduction mode for CE.

    Returns:
        Dict with keys:
        - `loss_caption`
        - `token_accuracy`
        - `num_valid_tokens`
    """
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction='{reduction}'.")
    if isinstance(ignore_index, bool) or not isinstance(ignore_index, int):
        raise TypeError("ignore_index must be an integer.")

    if not isinstance(decoder_logits, torch.Tensor):
        raise TypeError("decoder_logits must be torch.Tensor.")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be torch.Tensor.")

    if decoder_logits.ndim != 3:
        raise LossShapeError(
            f"decoder_logits must have shape [B,L,V], got {tuple(decoder_logits.shape)}."
        )
    if labels.ndim != 2:
        raise LossShapeError(f"labels must have shape [B,L], got {tuple(labels.shape)}.")

    batch_size: int = int(decoder_logits.shape[0])
    seq_len: int = int(decoder_logits.shape[1])
    vocab_size: int = int(decoder_logits.shape[2])

    if tuple(labels.shape) != (batch_size, seq_len):
        raise LossShapeError(
            "labels must align with decoder logits first two dims. "
            f"Got logits={tuple(decoder_logits.shape)}, labels={tuple(labels.shape)}."
        )
    if vocab_size <= 1:
        raise LossShapeError(f"decoder vocab size must be > 1, got {vocab_size}.")

    logits_flat: torch.Tensor = decoder_logits.reshape(batch_size * seq_len, vocab_size).to(dtype=torch.float32)
    labels_flat: torch.Tensor = labels.reshape(batch_size * seq_len).to(dtype=torch.long)

    loss_caption: torch.Tensor = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=int(ignore_index),
        reduction=reduction,
    )

    with torch.no_grad():
        valid_mask: torch.Tensor = labels_flat.ne(int(ignore_index))
        num_valid_tokens: torch.Tensor = valid_mask.to(dtype=torch.float32).sum()

        if float(num_valid_tokens.item()) > 0.0:
            preds: torch.Tensor = torch.argmax(logits_flat, dim=-1)
            correct: torch.Tensor = (preds.eq(labels_flat) & valid_mask).to(dtype=torch.float32).sum()
            token_accuracy: torch.Tensor = correct / num_valid_tokens.clamp_min(1.0)
        else:
            token_accuracy = torch.zeros((), device=decoder_logits.device, dtype=torch.float32)

    return {
        "loss_caption": loss_caption,
        "token_accuracy": token_accuracy,
        "num_valid_tokens": num_valid_tokens,
    }


def compute_multimodal_stage_loss(
    contrastive_outputs: Mapping[str, Any],
    caption_outputs: Mapping[str, Any],
    contrastive_weight: Optional[float] = None,
    caption_weight: Optional[float] = None,
    ignore_index: int = _DEFAULT_IGNORE_INDEX,
) -> Dict[str, torch.Tensor]:
    """Compute weighted stage-2/3 multimodal objective.

    This combines:
    - symmetric contrastive loss from `contrastive_outputs`
    - caption CE loss from `caption_outputs`

    Args:
        contrastive_outputs: Mapping from `CoCaModel.forward_contrastive`.
        caption_outputs: Mapping from `CoCaModel.forward_captioning`.
        contrastive_weight: Optional config weight. If None, defaults to 1.0.
        caption_weight: Optional config weight. If None, defaults to 1.0.
        ignore_index: Ignore index for caption loss.

    Returns:
        Dict containing:
        - `loss_total`
        - `loss_contrastive`
        - `loss_caption`
        - `loss_i2t`, `loss_t2i`, `acc_i2t`, `acc_t2i`
        - `token_accuracy`, `num_valid_tokens`
        - `contrastive_weight`, `caption_weight`
    """
    if not isinstance(contrastive_outputs, Mapping):
        raise TypeError("contrastive_outputs must be a mapping.")
    if not isinstance(caption_outputs, Mapping):
        raise TypeError("caption_outputs must be a mapping.")

    # Flexible extraction for compatibility with current CoCaModel outputs.
    logits_per_image: Optional[torch.Tensor] = _get_optional_tensor(
        mapping=contrastive_outputs,
        key="logits_per_image",
    )
    logits_per_text: Optional[torch.Tensor] = _get_optional_tensor(
        mapping=contrastive_outputs,
        key="logits_per_text",
    )

    image_embeddings: Optional[torch.Tensor] = _get_optional_tensor(
        mapping=contrastive_outputs,
        key="image_embeddings",
    )
    text_embeddings: Optional[torch.Tensor] = _get_optional_tensor(
        mapping=contrastive_outputs,
        key="text_embeddings",
    )
    logit_scale: Optional[torch.Tensor] = _get_optional_tensor(
        mapping=contrastive_outputs,
        key="logit_scale",
    )

    contrastive_dict: Dict[str, torch.Tensor] = contrastive_info_nce_loss(
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        logits_per_image=logits_per_image,
        logits_per_text=logits_per_text,
        logit_scale=logit_scale,
        normalize_embeddings=True,
        reduction="mean",
    )

    decoder_logits: torch.Tensor = _get_required_tensor(caption_outputs, "decoder_logits")
    labels: torch.Tensor = _get_required_tensor(caption_outputs, "labels")
    caption_dict: Dict[str, torch.Tensor] = caption_cross_entropy_loss(
        decoder_logits=decoder_logits,
        labels=labels,
        ignore_index=int(ignore_index),
        reduction="mean",
    )

    contrastive_weight_t: torch.Tensor = _resolve_weight_tensor(
        contrastive_weight,
        default=_DEFAULT_CONTRASTIVE_WEIGHT,
        device=contrastive_dict["loss_contrastive"].device,
        dtype=contrastive_dict["loss_contrastive"].dtype,
        name="contrastive_weight",
    )
    caption_weight_t: torch.Tensor = _resolve_weight_tensor(
        caption_weight,
        default=_DEFAULT_CAPTION_WEIGHT,
        device=caption_dict["loss_caption"].device,
        dtype=caption_dict["loss_caption"].dtype,
        name="caption_weight",
    )

    loss_total: torch.Tensor = (
        contrastive_weight_t * contrastive_dict["loss_contrastive"]
        + caption_weight_t * caption_dict["loss_caption"]
    )

    output: Dict[str, torch.Tensor] = {
        "loss_total": loss_total,
        "loss_contrastive": contrastive_dict["loss_contrastive"],
        "loss_caption": caption_dict["loss_caption"],
        "loss_i2t": contrastive_dict["loss_i2t"],
        "loss_t2i": contrastive_dict["loss_t2i"],
        "acc_i2t": contrastive_dict["acc_i2t"],
        "acc_t2i": contrastive_dict["acc_t2i"],
        "token_accuracy": caption_dict["token_accuracy"],
        "num_valid_tokens": caption_dict["num_valid_tokens"],
        "contrastive_weight": contrastive_weight_t,
        "caption_weight": caption_weight_t,
    }

    if not torch.isfinite(output["loss_total"]):
        raise LossesError("compute_multimodal_stage_loss produced non-finite total loss.")

    return output


def _get_required_tensor(mapping: Mapping[str, Any], key: str) -> torch.Tensor:
    value: Any = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise KeyError(f"Expected tensor key '{key}' in mapping.")
    return value


def _get_optional_tensor(mapping: Mapping[str, Any], key: str) -> Optional[torch.Tensor]:
    value: Any = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Key '{key}' must be torch.Tensor when provided.")
    return value


def _resolve_weight_tensor(
    weight: Optional[float],
    default: float,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if weight is None:
        weight_value: float = float(default)
    else:
        if not isinstance(weight, (int, float)):
            raise TypeError(f"{name} must be numeric or None.")
        weight_value = float(weight)

    if weight_value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {weight_value}.")

    return torch.tensor(weight_value, device=device, dtype=dtype)


def _to_btd(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to `[B,T,D]` shape."""
    if tensor.ndim == 4:
        # [B,H,W,D]
        batch_size: int = int(tensor.shape[0])
        h_tokens: int = int(tensor.shape[1])
        w_tokens: int = int(tensor.shape[2])
        feat_dim: int = int(tensor.shape[3])
        if feat_dim <= 0:
            raise LossShapeError("Invalid feature dimension in 4D tensor.")
        return tensor.reshape(batch_size, h_tokens * w_tokens, feat_dim)

    if tensor.ndim == 3:
        # [B,T,D]
        return tensor

    if tensor.ndim == 2:
        # Ambiguous between [T,D] and [B,T] for masks; this helper is for features.
        # We treat as [T,D] -> [1,T,D].
        return tensor.unsqueeze(0)

    raise LossShapeError(
        f"Expected rank 2/3/4 feature tensor, got rank={tensor.ndim} shape={tuple(tensor.shape)}."
    )


def _to_bt_mask(mask: torch.Tensor, batch_size: int, num_tokens: int) -> torch.Tensor:
    """Normalize mask to `[B,T]` with boolean dtype.

    Accepted forms:
    - [B,T]
    - [T] (broadcasted to batch)
    - [B,H,W] (flattened)
    - [H,W] (flattened and broadcasted)
    """
    if mask.ndim == 2:
        if tuple(mask.shape) == (batch_size, num_tokens):
            return mask.to(dtype=torch.bool)
        if int(mask.shape[0]) * int(mask.shape[1]) == num_tokens and batch_size == 1:
            return mask.reshape(1, num_tokens).to(dtype=torch.bool)
        raise LossShapeError(
            "2D mask shape mismatch. "
            f"Expected {(batch_size, num_tokens)} or {(1, num_tokens)} compatible reshape, got {tuple(mask.shape)}."
        )

    if mask.ndim == 1:
        if int(mask.shape[0]) != num_tokens:
            raise LossShapeError(
                f"1D mask length must be num_tokens={num_tokens}, got {int(mask.shape[0])}."
            )
        return mask.unsqueeze(0).expand(batch_size, -1).to(dtype=torch.bool)

    if mask.ndim == 3:
        if int(mask.shape[0]) != batch_size:
            raise LossShapeError(
                "3D mask batch mismatch. "
                f"Expected B={batch_size}, got shape={tuple(mask.shape)}."
            )
        flattened: torch.Tensor = mask.reshape(batch_size, -1)
        if int(flattened.shape[1]) != num_tokens:
            raise LossShapeError(
                f"Flattened 3D mask tokens must be {num_tokens}, got {int(flattened.shape[1])}."
            )
        return flattened.to(dtype=torch.bool)

    if mask.ndim == 4:
        # Rare case: [B,1,H,W] or [B,H,W,1]
        if int(mask.shape[0]) != batch_size:
            raise LossShapeError(
                "4D mask batch mismatch. "
                f"Expected B={batch_size}, got shape={tuple(mask.shape)}."
            )
        flattened = mask.reshape(batch_size, -1)
        if int(flattened.shape[1]) != num_tokens:
            raise LossShapeError(
                f"Flattened 4D mask tokens must be {num_tokens}, got {int(flattened.shape[1])}."
            )
        return flattened.to(dtype=torch.bool)

    raise LossShapeError(
        f"Unsupported mask rank={mask.ndim}, shape={tuple(mask.shape)}."
    )


def _dist_is_active() -> bool:
    return bool(dist.is_available() and dist.is_initialized())


__all__ = [
    "LossesError",
    "LossShapeError",
    "IBOTLoss",
    "contrastive_info_nce_loss",
    "caption_cross_entropy_loss",
    "compute_multimodal_stage_loss",
]
