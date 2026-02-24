"""Zero-shot slide classification for TITAN vision-language embeddings.

This module implements the design-locked evaluator interface:
- Evaluator.run_zero_shot(slide_emb: np.ndarray, class_text_emb: np.ndarray, y: np.ndarray) -> dict

Protocol alignment:
- CLIP-style zero-shot classification via cosine similarity in a shared space.
- L2-normalize slide and text embeddings before similarity scoring.
- Prediction rule: argmax_c <u_i, v_c>.
- Metrics:
  - balanced accuracy (all tasks)
  - AUROC (binary tasks)
  - weighted F1 (multiclass tasks)

Prompt-ensemble support:
- `class_text_emb` accepts either:
  - [C, D] pre-aggregated class embeddings
  - [C, P, D] prompt-level embeddings per class (aggregated internally)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DEFAULT_EPS: float = 1.0e-12


class ZeroShotError(RuntimeError):
    """Base exception for zero-shot evaluation failures."""


class ZeroShotSchemaError(ZeroShotError):
    """Raised when zero-shot inputs violate shape/type contracts."""


@dataclass(frozen=True)
class _PreparedInputs:
    """Normalized and validated zero-shot inputs."""

    slide_emb: np.ndarray  # [N, D], l2-normalized
    class_emb: np.ndarray  # [C, D], l2-normalized
    y: np.ndarray  # [N], integer class indices in [0, C-1]


class ZeroShotEvaluator:
    """Zero-shot evaluator with strict config-aligned contracts."""

    def __init__(self, eps: float = _DEFAULT_EPS) -> None:
        if not isinstance(eps, (int, float)):
            raise TypeError("eps must be numeric.")
        eps_value: float = float(eps)
        if eps_value <= 0.0:
            raise ValueError("eps must be > 0.")

        self.eps: float = eps_value

        # Provenance constants.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

    def run_zero_shot(self, slide_emb: np.ndarray, class_text_emb: np.ndarray, y: np.ndarray) -> dict:
        """Run zero-shot classification from slide and class text embeddings.

        Args:
            slide_emb: Slide embeddings of shape [N, 768].
            class_text_emb: Class text embeddings of shape [C, 768] or [C, P, 768].
            y: Ground-truth class indices aligned with `class_text_emb`, shape [N].

        Returns:
            Structured result dictionary with metrics, logits summary, and predictions.
        """
        prepared: _PreparedInputs = self._prepare_inputs(
            slide_emb=slide_emb,
            class_text_emb=class_text_emb,
            y=y,
        )

        logits: np.ndarray = np.matmul(prepared.slide_emb, prepared.class_emb.T)
        if logits.ndim != 2:
            raise ZeroShotSchemaError(
                f"Internal logits must be rank-2 [N,C], got {tuple(logits.shape)}."
            )

        y_pred: np.ndarray = np.argmax(logits, axis=1).astype(np.int64, copy=False)

        metrics: Dict[str, float] = self._compute_metrics(
            y_true=prepared.y,
            y_pred=y_pred,
            logits=logits,
        )

        result: Dict[str, Any] = {
            "task": "zero_shot",
            "protocol": {
                "scoring": "cosine_similarity_via_l2_normalized_dot_product",
                "prediction_rule": "argmax_class_similarity",
                "prompt_ensemble_supported": True,
                "tie_break": "lowest_class_index_via_argmax_first_occurrence",
            },
            "input": {
                "n_samples": int(prepared.slide_emb.shape[0]),
                "n_classes": int(prepared.class_emb.shape[0]),
                "embedding_dim": int(prepared.slide_emb.shape[1]),
                "class_text_emb_rank": int(np.asarray(class_text_emb).ndim),
            },
            "metrics": metrics,
            "outputs": {
                "y_pred": y_pred.astype(np.int64, copy=False).tolist(),
                "max_similarity": np.max(logits, axis=1).astype(np.float64, copy=False).tolist(),
            },
            "diagnostics": {
                "logits_min": float(np.min(logits)),
                "logits_max": float(np.max(logits)),
                "logits_mean": float(np.mean(logits)),
                "logits_std": float(np.std(logits, ddof=0)),
            },
        }
        return result

    def _prepare_inputs(
        self,
        slide_emb: Any,
        class_text_emb: Any,
        y: Any,
    ) -> _PreparedInputs:
        slide_np: np.ndarray = self._validate_slide_embeddings(slide_emb)
        class_np: np.ndarray = self._validate_and_aggregate_class_embeddings(class_text_emb)
        y_np: np.ndarray = self._validate_labels(y=y, n_samples=int(slide_np.shape[0]), n_classes=int(class_np.shape[0]))

        slide_norm: np.ndarray = self._l2_normalize_rows(slide_np)
        class_norm: np.ndarray = self._l2_normalize_rows(class_np)

        if not np.isfinite(slide_norm).all():
            raise ZeroShotSchemaError("Normalized slide embeddings contain NaN/Inf.")
        if not np.isfinite(class_norm).all():
            raise ZeroShotSchemaError("Normalized class embeddings contain NaN/Inf.")

        return _PreparedInputs(
            slide_emb=slide_norm,
            class_emb=class_norm,
            y=y_np,
        )

    def _validate_slide_embeddings(self, slide_emb: Any) -> np.ndarray:
        if not isinstance(slide_emb, np.ndarray):
            slide_emb = np.asarray(slide_emb)

        x: np.ndarray = np.asarray(slide_emb, dtype=np.float64)
        if x.ndim != 2:
            raise ZeroShotSchemaError(f"slide_emb must be rank-2 [N,D], got {tuple(x.shape)}.")
        if int(x.shape[0]) <= 0:
            raise ZeroShotSchemaError("slide_emb must contain at least one sample.")
        if int(x.shape[1]) != self.feature_dim:
            raise ZeroShotSchemaError(
                f"slide_emb second dimension must be {self.feature_dim}, got {int(x.shape[1])}."
            )
        if not np.isfinite(x).all():
            raise ZeroShotSchemaError("slide_emb contains NaN/Inf values.")
        return x.astype(np.float32, copy=False)

    def _validate_and_aggregate_class_embeddings(self, class_text_emb: Any) -> np.ndarray:
        if not isinstance(class_text_emb, np.ndarray):
            class_text_emb = np.asarray(class_text_emb)

        arr: np.ndarray = np.asarray(class_text_emb, dtype=np.float64)
        if arr.ndim == 2:
            # [C, D]
            if int(arr.shape[0]) <= 1:
                raise ZeroShotSchemaError("class_text_emb must contain at least 2 classes.")
            if int(arr.shape[1]) != self.feature_dim:
                raise ZeroShotSchemaError(
                    f"class_text_emb second dimension must be {self.feature_dim}, got {int(arr.shape[1])}."
                )
            if not np.isfinite(arr).all():
                raise ZeroShotSchemaError("class_text_emb contains NaN/Inf values.")
            return arr.astype(np.float32, copy=False)

        if arr.ndim == 3:
            # [C, P, D] prompt ensemble. Aggregate deterministically.
            num_classes: int = int(arr.shape[0])
            num_prompts: int = int(arr.shape[1])
            emb_dim: int = int(arr.shape[2])

            if num_classes <= 1:
                raise ZeroShotSchemaError("class_text_emb must contain at least 2 classes.")
            if num_prompts <= 0:
                raise ZeroShotSchemaError("Prompt dimension P must be > 0 for class_text_emb rank-3.")
            if emb_dim != self.feature_dim:
                raise ZeroShotSchemaError(
                    f"class_text_emb last dimension must be {self.feature_dim}, got {emb_dim}."
                )
            if not np.isfinite(arr).all():
                raise ZeroShotSchemaError("class_text_emb contains NaN/Inf values.")

            prompt_norm: np.ndarray = self._l2_normalize_last_dim(arr.astype(np.float32, copy=False))
            class_mean: np.ndarray = np.mean(prompt_norm, axis=1, dtype=np.float64).astype(np.float32, copy=False)
            class_agg: np.ndarray = self._l2_normalize_rows(class_mean)
            return class_agg

        raise ZeroShotSchemaError(
            "class_text_emb must have shape [C,D] or [C,P,D]. "
            f"Got rank={arr.ndim} shape={tuple(arr.shape)}."
        )

    def _validate_labels(self, y: Any, n_samples: int, n_classes: int) -> np.ndarray:
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        labels: np.ndarray = np.asarray(y)
        if labels.ndim != 1:
            raise ZeroShotSchemaError(f"y must be rank-1 [N], got {tuple(labels.shape)}.")
        if int(labels.shape[0]) != int(n_samples):
            raise ZeroShotSchemaError(
                f"y length must match slide_emb rows ({n_samples}), got {int(labels.shape[0])}."
            )

        # Enforce class-index labels aligned to class_text_emb rows.
        if np.issubdtype(labels.dtype, np.integer):
            y_idx: np.ndarray = labels.astype(np.int64, copy=False)
        elif np.issubdtype(labels.dtype, np.floating):
            if not np.all(np.equal(labels, np.floor(labels))):
                raise ZeroShotSchemaError(
                    "Floating labels must be integer-valued class indices aligned to class_text_emb."
                )
            y_idx = labels.astype(np.int64, copy=False)
        else:
            raise ZeroShotSchemaError(
                "y must contain integer class indices aligned to class_text_emb rows. "
                f"Got dtype={labels.dtype}."
            )

        if np.any(y_idx < 0) or np.any(y_idx >= int(n_classes)):
            raise ZeroShotSchemaError(
                f"Label indices out of range [0, {n_classes - 1}] for class_text_emb size={n_classes}."
            )

        unique_classes: np.ndarray = np.unique(y_idx)
        if int(unique_classes.shape[0]) < 2:
            raise ZeroShotSchemaError("y must contain at least 2 classes for zero-shot evaluation.")

        return y_idx

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

        unique_classes: np.ndarray = np.unique(y_true)
        n_classes_present: int = int(unique_classes.shape[0])

        if n_classes_present == 2:
            # Positive class is max class index among present classes.
            pos_class: int = int(np.max(unique_classes))
            if int(logits.shape[1]) <= pos_class:
                raise ZeroShotSchemaError(
                    "Cannot compute binary AUROC: logits do not contain positive-class column. "
                    f"logits_shape={tuple(logits.shape)}, pos_class={pos_class}."
                )
            y_score: np.ndarray = logits[:, pos_class]
            metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        else:
            metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))

        return metrics

    def _l2_normalize_rows(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2:
            raise ZeroShotSchemaError(f"Row normalization expects rank-2 array, got {tuple(x.shape)}.")
        norms: np.ndarray = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, float(self.eps)).astype(np.float32, copy=False)
        out: np.ndarray = x.astype(np.float32, copy=False) / norms
        return out

    def _l2_normalize_last_dim(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3:
            raise ZeroShotSchemaError(
                f"Last-dim normalization expects rank-3 array, got {tuple(x.shape)}."
            )
        norms: np.ndarray = np.linalg.norm(x, ord=2, axis=2, keepdims=True)
        norms = np.maximum(norms, float(self.eps)).astype(np.float32, copy=False)
        out: np.ndarray = x.astype(np.float32, copy=False) / norms
        return out


# Convenience functional API.
def run_zero_shot(slide_emb: np.ndarray, class_text_emb: np.ndarray, y: np.ndarray) -> dict:
    """Run zero-shot evaluation with default, config-aligned settings."""
    evaluator: ZeroShotEvaluator = ZeroShotEvaluator()
    return evaluator.run_zero_shot(slide_emb=slide_emb, class_text_emb=class_text_emb, y=y)


__all__ = [
    "ZeroShotError",
    "ZeroShotSchemaError",
    "ZeroShotEvaluator",
    "run_zero_shot",
]
