"""k-NN and prototype probing for TITAN slide embeddings.

This module implements the design-locked probe interface:
- Evaluator.run_knn_probe(features: np.ndarray, y: np.ndarray, k: int = 20) -> dict

Implemented behavior:
- Shared preprocessing: center + L2 normalization.
- Nonparametric probes:
  - Prototype probe (SimpleShot-style nearest class prototype).
  - k-NN probe with deterministic majority voting.
- Metrics:
  - Balanced accuracy (always)
  - Weighted F1 (multiclass)
  - AUROC (binary when score estimates are available)

Notes:
- The public method name is preserved via ``KNNProbeEvaluator.run_knn_probe``.
- A convenience function ``run_knn_probe`` is also exported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_KNN_DEFAULT_K: int = 20
_KNN_DISTANCE: str = "euclidean"
_KNN_PREPROCESS: Tuple[str, str] = ("center", "l2_normalize")

_DEFAULT_EPS: float = 1.0e-12


class KNNProbeError(RuntimeError):
    """Base exception for k-NN probe failures."""


class KNNProbeSchemaError(KNNProbeError):
    """Raised when input schema/shape contracts are violated."""


@dataclass(frozen=True)
class _ProbeOutputs:
    """Container for one probe mode outputs."""

    y_pred: np.ndarray
    y_score: Optional[np.ndarray]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]


class KNNProbeEvaluator:
    """Nonparametric embedding evaluator with prototype + k-NN probes.

    Args:
        default_k: Default neighbor count. Must be 20 (config-bound).
        distance: Distance metric. Must be ``euclidean``.
        preprocess: Preprocessing sequence. Must include ``center`` and
            ``l2_normalize``.
        embedding_service: Optional object exposing
            ``center_and_l2(x, mean_vec=None)``.
        allow_self_match: If False, uses leave-one-out behavior when query and
            reference are the same pool.
    """

    def __init__(
        self,
        default_k: int = _KNN_DEFAULT_K,
        distance: str = _KNN_DISTANCE,
        preprocess: Sequence[str] = _KNN_PREPROCESS,
        embedding_service: Optional[Any] = None,
        allow_self_match: bool = False,
    ) -> None:
        if isinstance(default_k, bool) or not isinstance(default_k, int):
            raise TypeError("default_k must be an integer.")
        if default_k <= 0:
            raise ValueError("default_k must be > 0.")

        distance_value: str = str(distance).strip().lower()
        if distance_value != _KNN_DISTANCE:
            raise ValueError(
                f"distance must be '{_KNN_DISTANCE}' per config, got '{distance}'."
            )

        preprocess_tuple: Tuple[str, ...] = tuple(str(item).strip() for item in preprocess)
        required: set[str] = set(_KNN_PREPROCESS)
        if not required.issubset(set(preprocess_tuple)):
            raise ValueError(
                "preprocess must include center and l2_normalize. "
                f"Got {preprocess_tuple}."
            )

        # Strict config binding.
        if int(default_k) != _KNN_DEFAULT_K:
            raise ValueError(
                f"default_k must be {_KNN_DEFAULT_K}, got {default_k}."
            )

        if embedding_service is not None:
            center_fn: Any = getattr(embedding_service, "center_and_l2", None)
            if not callable(center_fn):
                raise TypeError(
                    "embedding_service must implement callable center_and_l2(x, mean_vec=None)."
                )

        self.default_k: int = int(default_k)
        self.distance: str = distance_value
        self.preprocess: Tuple[str, ...] = preprocess_tuple
        self.embedding_service: Optional[Any] = embedding_service
        self.allow_self_match: bool = bool(allow_self_match)

        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

    def run_knn_probe(self, features: np.ndarray, y: np.ndarray, k: int = _KNN_DEFAULT_K) -> dict:
        """Run prototype and k-NN probes on one embedding set.

        This method uses leave-one-out query/reference evaluation when
        ``allow_self_match=False`` (default), preventing trivial self-retrieval.

        Args:
            features: Embeddings of shape [N, 768].
            y: Labels of shape [N].
            k: Neighbor count (default 20).

        Returns:
            Structured dictionary with protocol metadata and probe metrics.
        """
        x: np.ndarray = self._validate_features(features)
        labels_raw: np.ndarray = self._validate_labels(y, expected_n=int(x.shape[0]))

        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k <= 0:
            raise ValueError("k must be > 0.")

        encoder: LabelEncoder = LabelEncoder()
        labels_encoded: np.ndarray = encoder.fit_transform(labels_raw)
        classes: np.ndarray = np.asarray(encoder.classes_)
        n_samples: int = int(x.shape[0])
        n_classes: int = int(classes.shape[0])

        if n_classes < 2:
            raise KNNProbeSchemaError("At least two classes are required for k-NN probe.")

        # Shared preprocessing: center + L2 normalize.
        x_norm, mean_vec = self._center_and_l2(x=x)

        # Maximum valid k under leave-one-out semantics.
        max_k: int = n_samples if self.allow_self_match else n_samples - 1
        if max_k <= 0:
            raise KNNProbeSchemaError(
                "Insufficient samples for leave-one-out probing (need at least 2)."
            )
        effective_k: int = min(int(k), int(max_k))

        prototype_outputs: _ProbeOutputs = self._run_prototype_probe_loo(
            x=x_norm,
            y=labels_encoded,
            n_classes=n_classes,
        )
        knn_outputs: _ProbeOutputs = self._run_knn_probe_loo(
            x=x_norm,
            y=labels_encoded,
            n_classes=n_classes,
            k=effective_k,
        )

        output: Dict[str, Any] = {
            "task": "knn_probe",
            "input": {
                "n_samples": int(n_samples),
                "n_features": int(x.shape[1]),
                "n_classes": int(n_classes),
                "classes": [str(item) for item in classes.tolist()],
            },
            "protocol": {
                "distance": self.distance,
                "preprocess": list(self.preprocess),
                "default_k": int(self.default_k),
                "requested_k": int(k),
                "effective_k": int(effective_k),
                "allow_self_match": bool(self.allow_self_match),
            },
            "preprocess": {
                "mean_vec_shape": [int(mean_vec.shape[0])],
            },
            "prototype": {
                "metrics": dict(prototype_outputs.metrics),
                "metadata": dict(prototype_outputs.metadata),
            },
            "knn": {
                "metrics": dict(knn_outputs.metrics),
                "metadata": dict(knn_outputs.metadata),
            },
        }
        return output

    def _run_knn_probe_loo(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        k: int,
    ) -> _ProbeOutputs:
        n_samples: int = int(x.shape[0])

        # Pairwise Euclidean distances.
        dist_matrix: np.ndarray = self._pairwise_euclidean(x, x)

        if not self.allow_self_match:
            np.fill_diagonal(dist_matrix, np.inf)

        if k <= 0:
            raise KNNProbeSchemaError(f"k must be > 0, got {k}.")
        if k > n_samples:
            raise KNNProbeSchemaError(
                f"k cannot exceed n_samples={n_samples}, got {k}."
            )

        # Top-k neighbor indices per query.
        neighbor_idx: np.ndarray = np.argpartition(dist_matrix, kth=k - 1, axis=1)[:, :k]

        # Deterministic ordering within top-k by (distance, index).
        sorted_neighbor_idx: np.ndarray = np.zeros_like(neighbor_idx)
        for row in range(n_samples):
            row_idx: np.ndarray = neighbor_idx[row]
            row_dist: np.ndarray = dist_matrix[row, row_idx]
            order: np.ndarray = np.lexsort((row_idx, row_dist))
            sorted_neighbor_idx[row] = row_idx[order]

        y_pred: np.ndarray = np.zeros((n_samples,), dtype=np.int64)
        y_score: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)

        tie_count: int = 0
        for row in range(n_samples):
            neigh_indices: np.ndarray = sorted_neighbor_idx[row]
            neigh_labels: np.ndarray = y[neigh_indices]
            neigh_distances: np.ndarray = dist_matrix[row, neigh_indices]

            counts: np.ndarray = np.bincount(neigh_labels, minlength=n_classes).astype(np.int64)
            y_score[row, :] = counts.astype(np.float64) / float(k)

            max_count: int = int(np.max(counts))
            candidates: np.ndarray = np.where(counts == max_count)[0]

            if int(candidates.shape[0]) == 1:
                winner: int = int(candidates[0])
            else:
                tie_count += 1
                # Deterministic tie-break:
                # 1) smallest mean distance among tied classes
                # 2) smallest class index
                mean_dist_by_class: Dict[int, float] = {}
                for cls_id in candidates.tolist():
                    cls_mask: np.ndarray = neigh_labels == int(cls_id)
                    cls_dist: np.ndarray = neigh_distances[cls_mask]
                    if cls_dist.size == 0:
                        mean_dist_by_class[int(cls_id)] = float(np.inf)
                    else:
                        mean_dist_by_class[int(cls_id)] = float(np.mean(cls_dist))

                sorted_candidates: List[int] = sorted(
                    [int(v) for v in candidates.tolist()],
                    key=lambda class_id: (mean_dist_by_class[class_id], class_id),
                )
                winner = int(sorted_candidates[0])

            y_pred[row] = winner

        metrics: Dict[str, float] = self._compute_metrics(
            y_true=y,
            y_pred=y_pred,
            y_score=y_score,
        )

        metadata: Dict[str, Any] = {
            "mode": "knn",
            "k": int(k),
            "tie_count": int(tie_count),
        }

        return _ProbeOutputs(
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            metadata=metadata,
        )

    def _run_prototype_probe_loo(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_classes: int,
    ) -> _ProbeOutputs:
        n_samples: int = int(x.shape[0])
        feat_dim: int = int(x.shape[1])

        # Precompute sums/counts per class for efficient leave-one-out prototypes.
        class_sums: np.ndarray = np.zeros((n_classes, feat_dim), dtype=np.float64)
        class_counts: np.ndarray = np.zeros((n_classes,), dtype=np.int64)
        for idx in range(n_samples):
            class_id: int = int(y[idx])
            class_sums[class_id] += x[idx]
            class_counts[class_id] += 1

        y_pred: np.ndarray = np.zeros((n_samples,), dtype=np.int64)
        y_score: np.ndarray = np.zeros((n_samples, n_classes), dtype=np.float64)

        for query_idx in range(n_samples):
            query_vec: np.ndarray = x[query_idx]
            query_class: int = int(y[query_idx])

            prototypes: np.ndarray = np.zeros((n_classes, feat_dim), dtype=np.float64)
            valid_proto: np.ndarray = np.ones((n_classes,), dtype=np.bool_)

            for class_id in range(n_classes):
                count_value: int = int(class_counts[class_id])
                sum_value: np.ndarray = class_sums[class_id]

                if (not self.allow_self_match) and class_id == query_class:
                    count_value -= 1
                    sum_value = sum_value - query_vec

                if count_value <= 0:
                    # Fallback to global class prototype if leave-one-out empty.
                    count_fallback: int = int(class_counts[class_id])
                    if count_fallback <= 0:
                        valid_proto[class_id] = False
                        continue
                    proto_vec: np.ndarray = class_sums[class_id] / float(count_fallback)
                else:
                    proto_vec = sum_value / float(count_value)

                # Keep prototype in centered space and L2 normalize for stable distance.
                norm_value: float = float(np.linalg.norm(proto_vec, ord=2))
                if norm_value <= _DEFAULT_EPS:
                    valid_proto[class_id] = False
                    continue
                prototypes[class_id] = proto_vec / norm_value

            if not np.any(valid_proto):
                raise KNNProbeSchemaError("No valid class prototypes available for query.")

            # Distances to prototypes (invalid prototypes set to +inf).
            dists: np.ndarray = np.full((n_classes,), np.inf, dtype=np.float64)
            valid_indices: np.ndarray = np.where(valid_proto)[0]
            valid_protos: np.ndarray = prototypes[valid_indices]
            dists_valid: np.ndarray = np.linalg.norm(valid_protos - query_vec.reshape(1, -1), axis=1)
            dists[valid_indices] = dists_valid

            pred_class: int = int(np.argmin(dists))
            y_pred[query_idx] = pred_class

            # Convert negative distances to probability-like scores via softmax.
            logits: np.ndarray = -dists
            logits = logits - np.max(logits[np.isfinite(logits)])
            probs: np.ndarray = np.zeros((n_classes,), dtype=np.float64)
            finite_mask: np.ndarray = np.isfinite(logits)
            exp_logits: np.ndarray = np.exp(logits[finite_mask])
            probs[finite_mask] = exp_logits / np.sum(exp_logits)
            y_score[query_idx, :] = probs

        metrics: Dict[str, float] = self._compute_metrics(
            y_true=y,
            y_pred=y_pred,
            y_score=y_score,
        )

        metadata: Dict[str, Any] = {
            "mode": "prototype",
            "leave_one_out": bool(not self.allow_self_match),
        }

        return _ProbeOutputs(
            y_pred=y_pred,
            y_score=y_score,
            metrics=metrics,
            metadata=metadata,
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        unique_classes: np.ndarray = np.unique(y_true)
        n_classes: int = int(unique_classes.shape[0])

        if n_classes == 2:
            if y_score is not None:
                if y_score.ndim != 2 or int(y_score.shape[1]) < 2:
                    raise KNNProbeSchemaError(
                        f"Binary score matrix must have shape [N,2], got {tuple(y_score.shape)}."
                    )
                positive_scores: np.ndarray = y_score[:, 1]
                metrics["auroc"] = float(roc_auc_score(y_true, positive_scores))
        else:
            metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))

        return metrics

    def _center_and_l2(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.embedding_service is not None:
            centered, mean_vec = self.embedding_service.center_and_l2(x=x, mean_vec=None)
            centered_np: np.ndarray = np.asarray(centered, dtype=np.float32)
            mean_np: np.ndarray = np.asarray(mean_vec, dtype=np.float32)
            if centered_np.shape != x.shape:
                raise KNNProbeSchemaError(
                    "embedding_service.center_and_l2 returned invalid shape. "
                    f"Expected {x.shape}, got {centered_np.shape}."
                )
            if mean_np.shape != (x.shape[1],):
                raise KNNProbeSchemaError(
                    "embedding_service.center_and_l2 returned invalid mean shape. "
                    f"Expected {(x.shape[1],)}, got {mean_np.shape}."
                )
            return centered_np, mean_np

        # Local fallback mirrors EmbeddingService.center_and_l2 behavior.
        mean_vec_local: np.ndarray = np.mean(x, axis=0, dtype=np.float64).astype(np.float32, copy=False)
        centered_local: np.ndarray = x.astype(np.float32, copy=False) - mean_vec_local.reshape(1, -1)
        norms: np.ndarray = np.linalg.norm(centered_local, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, float(_DEFAULT_EPS)).astype(np.float32, copy=False)
        normalized: np.ndarray = centered_local / norms

        if not np.isfinite(normalized).all():
            raise KNNProbeSchemaError("center_and_l2 produced NaN/Inf values.")

        return normalized.astype(np.float32, copy=False), mean_vec_local

    @staticmethod
    def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim != 2 or b.ndim != 2:
            raise KNNProbeSchemaError(
                f"pairwise_euclidean expects rank-2 inputs, got {a.shape} and {b.shape}."
            )
        if int(a.shape[1]) != int(b.shape[1]):
            raise KNNProbeSchemaError(
                "pairwise_euclidean feature dim mismatch: "
                f"{a.shape[1]} vs {b.shape[1]}."
            )

        # Numerically stable ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        a2: np.ndarray = np.sum(a * a, axis=1, keepdims=True)
        b2: np.ndarray = np.sum(b * b, axis=1, keepdims=True).T
        sq: np.ndarray = a2 + b2 - (2.0 * np.matmul(a, b.T))
        sq = np.maximum(sq, 0.0)
        dist: np.ndarray = np.sqrt(sq, dtype=np.float64)
        return dist

    @staticmethod
    def _validate_features(features: Any) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        x: np.ndarray = np.asarray(features, dtype=np.float64)

        if x.ndim != 2:
            raise KNNProbeSchemaError(f"features must be rank-2 [N,D], got {tuple(x.shape)}.")
        if int(x.shape[0]) <= 1:
            raise KNNProbeSchemaError("features must contain at least 2 samples.")
        if int(x.shape[1]) != _FEATURE_DIM:
            raise KNNProbeSchemaError(
                f"features second dimension must be {_FEATURE_DIM}, got {int(x.shape[1])}."
            )
        if not np.isfinite(x).all():
            raise KNNProbeSchemaError("features contain NaN/Inf values.")
        return x

    @staticmethod
    def _validate_labels(y: Any, expected_n: int) -> np.ndarray:
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        labels: np.ndarray = np.asarray(y)

        if labels.ndim != 1:
            raise KNNProbeSchemaError(f"y must be rank-1 [N], got {tuple(labels.shape)}.")
        if int(labels.shape[0]) != int(expected_n):
            raise KNNProbeSchemaError(
                f"y length must match number of samples ({expected_n}), got {int(labels.shape[0])}."
            )
        return labels


def run_knn_probe(features: np.ndarray, y: np.ndarray, k: int = _KNN_DEFAULT_K) -> dict:
    """Convenience functional API for k-NN/prototype probing."""
    evaluator: KNNProbeEvaluator = KNNProbeEvaluator()
    return evaluator.run_knn_probe(features=features, y=y, k=k)


__all__ = [
    "KNNProbeError",
    "KNNProbeSchemaError",
    "KNNProbeEvaluator",
    "run_knn_probe",
]
