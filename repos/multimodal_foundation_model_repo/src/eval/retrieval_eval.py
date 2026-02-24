"""Retrieval evaluator for THREADS downstream retrieval tasks.

This module implements the design-locked public interface:
- ``RetrievalEvaluator.__init__(metric: str, top_k: list[int]) -> None``
- ``RetrievalEvaluator.build_index(x_ref: object, y_ref: object) -> None``
- ``RetrievalEvaluator.query(x_q: object) -> object``
- ``RetrievalEvaluator.map_at_k(y_q: object, retrieved_labels: object, k: int) -> float``

Paper/config alignment:
- Similarity metric: L2 distance in embedding space.
- Retrieval cutoffs: mAP@1, mAP@5, mAP@10 (default top_k values).
- Embeddings are expected to be fixed-width vectors (THREADS defaults to 1024),
  but this evaluator validates shape consistency rather than hardcoding dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_METRIC: str = "l2"
DEFAULT_TOP_K: Tuple[int, ...] = (1, 5, 10)

DEFAULT_VALIDATE_NUMERICS: bool = True
DEFAULT_EPS: float = 1.0e-12

_ALLOWED_METRICS: Tuple[str, ...] = ("l2",)


class RetrievalEvaluatorError(Exception):
    """Base exception for retrieval evaluator failures."""


class RetrievalEvaluatorConfigError(RetrievalEvaluatorError):
    """Raised when evaluator configuration is invalid."""


class RetrievalEvaluatorInputError(RetrievalEvaluatorError):
    """Raised when build/query/metric inputs are malformed."""


class RetrievalEvaluatorNotReadyError(RetrievalEvaluatorError):
    """Raised when querying/scoring before building index."""


@dataclass(frozen=True)
class _IndexState:
    """Internal retrieval index state."""

    is_built: bool
    n_reference: int
    n_features: int


class RetrievalEvaluator:
    """L2 nearest-neighbor retrieval evaluator with mAP@k scoring."""

    def __init__(
        self,
        metric: str = DEFAULT_METRIC,
        top_k: list[int] | Tuple[int, ...] = DEFAULT_TOP_K,
    ) -> None:
        """Initialize retrieval evaluator.

        Args:
            metric: Distance metric key. Only ``"l2"`` is supported.
            top_k: Retrieval cutoffs. Defaults to ``[1, 5, 10]``.
        """
        self._metric: str = self._validate_metric(metric)
        self._top_k: Tuple[int, ...] = self._validate_top_k(top_k)
        self._max_k: int = int(max(self._top_k))

        self._x_ref: Optional[np.ndarray] = None
        self._y_ref: Optional[np.ndarray] = None
        self._ref_squared_norm: Optional[np.ndarray] = None
        self._ref_index: Optional[np.ndarray] = None

        self._state: _IndexState = _IndexState(
            is_built=False,
            n_reference=0,
            n_features=0,
        )

        self._validate_numerics: bool = DEFAULT_VALIDATE_NUMERICS

    def build_index(self, x_ref: object, y_ref: object) -> None:
        """Build retrieval reference index.

        Args:
            x_ref: Reference embedding matrix-like object with shape ``[N_ref, D]``.
            y_ref: Reference labels with shape ``[N_ref]``.
        """
        x_ref_array: np.ndarray = self._coerce_embeddings(x_ref, name="x_ref")
        y_ref_array: np.ndarray = self._coerce_labels(y_ref, name="y_ref")

        if int(x_ref_array.shape[0]) != int(y_ref_array.shape[0]):
            raise RetrievalEvaluatorInputError(
                "x_ref and y_ref sample counts mismatch: "
                f"{int(x_ref_array.shape[0])} vs {int(y_ref_array.shape[0])}."
            )

        if int(x_ref_array.shape[0]) < 1:
            raise RetrievalEvaluatorInputError("Reference index cannot be empty.")

        if self._max_k > int(x_ref_array.shape[0]):
            raise RetrievalEvaluatorInputError(
                "max(top_k) exceeds number of reference samples: "
                f"max_k={self._max_k}, n_reference={int(x_ref_array.shape[0])}."
            )

        self._x_ref = x_ref_array
        self._y_ref = y_ref_array
        self._ref_squared_norm = np.sum(np.square(x_ref_array), axis=1, dtype=np.float64)
        self._ref_index = np.arange(int(x_ref_array.shape[0]), dtype=np.int64)

        self._state = _IndexState(
            is_built=True,
            n_reference=int(x_ref_array.shape[0]),
            n_features=int(x_ref_array.shape[1]),
        )

    def query(self, x_q: object) -> object:
        """Query nearest neighbors for each embedding.

        Args:
            x_q: Query embedding matrix-like object with shape ``[N_q, D]``.

        Returns:
            Mapping with keys:
            - ``retrieved_indices``: ``[N_q, max_k]`` int64
            - ``retrieved_labels``: ``[N_q, max_k]`` labels
            - ``retrieved_distances``: ``[N_q, max_k]`` float64
            - ``metric``: retrieval metric name
            - ``top_k``: configured cutoffs
        """
        self._require_index_built()

        x_q_array: np.ndarray = self._coerce_embeddings(x_q, name="x_q")
        if int(x_q_array.shape[1]) != int(self._state.n_features):
            raise RetrievalEvaluatorInputError(
                "x_q feature width mismatch: "
                f"expected {self._state.n_features}, got {int(x_q_array.shape[1])}."
            )

        n_query: int = int(x_q_array.shape[0])
        max_k: int = int(self._max_k)

        retrieved_indices: np.ndarray = np.zeros((n_query, max_k), dtype=np.int64)
        retrieved_distances: np.ndarray = np.zeros((n_query, max_k), dtype=np.float64)

        x_ref_array: np.ndarray = self._x_ref if self._x_ref is not None else np.empty((0, 0), dtype=np.float64)
        ref_norm: np.ndarray = (
            self._ref_squared_norm if self._ref_squared_norm is not None else np.empty((0,), dtype=np.float64)
        )
        ref_index: np.ndarray = self._ref_index if self._ref_index is not None else np.empty((0,), dtype=np.int64)

        query_index: int
        for query_index in range(n_query):
            query_vector: np.ndarray = x_q_array[query_index]
            query_norm: float = float(np.dot(query_vector, query_vector))

            # Squared L2 distance: ||r-q||^2 = ||r||^2 + ||q||^2 - 2 r·q.
            dot_product: np.ndarray = np.matmul(x_ref_array, query_vector)
            distance_squared: np.ndarray = ref_norm + query_norm - 2.0 * dot_product
            distance_squared = np.maximum(distance_squared, 0.0)

            if self._validate_numerics and not np.isfinite(distance_squared).all():
                raise RetrievalEvaluatorInputError(
                    f"Non-finite distances encountered at query index={query_index}."
                )

            # Stable tie-break: primary key distance, secondary key reference index.
            rank_order: np.ndarray = np.lexsort((ref_index, distance_squared))
            top_indices: np.ndarray = rank_order[:max_k]

            retrieved_indices[query_index, :] = top_indices
            retrieved_distances[query_index, :] = np.sqrt(distance_squared[top_indices], dtype=np.float64)

        y_ref_array: np.ndarray = self._y_ref if self._y_ref is not None else np.empty((0,), dtype=object)
        retrieved_labels: np.ndarray = y_ref_array[retrieved_indices]

        output: dict[str, object] = {
            "retrieved_indices": retrieved_indices,
            "retrieved_labels": retrieved_labels,
            "retrieved_distances": retrieved_distances,
            "metric": self._metric,
            "top_k": list(self._top_k),
        }
        return output

    def map_at_k(self, y_q: object, retrieved_labels: object, k: int) -> float:
        """Compute mean average precision at cutoff ``k``.

        Formula (paper-aligned):
        - For each query i, ``AP_i@k = (1/k) * sum_{j=1..k} P_i(j) * rel_i(j)``
        - ``mAP@k = mean_i AP_i@k``
        where ``rel_i(j)=1`` when retrieved label at rank j matches query label.

        Args:
            y_q: Query labels of shape ``[N_q]``.
            retrieved_labels: Either:
                - rank-2 labels array ``[N_q, >=k]``
                - or mapping containing key ``"retrieved_labels"``.
            k: Cutoff rank.

        Returns:
            mAP@k value.
        """
        self._require_index_built()

        y_q_array: np.ndarray = self._coerce_labels(y_q, name="y_q")
        retrieved_label_array: np.ndarray = self._coerce_retrieved_labels(
            retrieved_labels,
            name="retrieved_labels",
        )

        k_int: int = self._validate_k(k)
        if k_int > int(retrieved_label_array.shape[1]):
            raise RetrievalEvaluatorInputError(
                "Requested k exceeds retrieved label width: "
                f"k={k_int}, retrieved_width={int(retrieved_label_array.shape[1])}."
            )

        if int(y_q_array.shape[0]) != int(retrieved_label_array.shape[0]):
            raise RetrievalEvaluatorInputError(
                "y_q and retrieved_labels query count mismatch: "
                f"{int(y_q_array.shape[0])} vs {int(retrieved_label_array.shape[0])}."
            )

        if int(y_q_array.shape[0]) == 0:
            raise RetrievalEvaluatorInputError("Cannot compute mAP@k with zero queries.")

        top_k_labels: np.ndarray = retrieved_label_array[:, :k_int]

        # Relevance matrix rel[i, j] = 1 if retrieved label at rank j matches query label.
        relevance: np.ndarray = (top_k_labels == y_q_array.reshape(-1, 1)).astype(np.float64)

        # Precision@j per query.
        cumulative_relevance: np.ndarray = np.cumsum(relevance, axis=1, dtype=np.float64)
        rank_positions: np.ndarray = np.arange(1, k_int + 1, dtype=np.float64).reshape(1, -1)
        precision_at_rank: np.ndarray = cumulative_relevance / rank_positions

        # AP_i@k = (1/k) * sum_j precision@j * relevance@j
        ap_per_query: np.ndarray = np.sum(precision_at_rank * relevance, axis=1, dtype=np.float64) / float(k_int)
        map_value: float = float(np.mean(ap_per_query, dtype=np.float64))

        if self._validate_numerics and not np.isfinite(map_value):
            raise RetrievalEvaluatorInputError("Computed mAP@k is non-finite.")

        return map_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_index_built(self) -> None:
        if not self._state.is_built:
            raise RetrievalEvaluatorNotReadyError("Index is not built. Call build_index() first.")

    def _coerce_embeddings(self, value: object, name: str) -> np.ndarray:
        array_value: np.ndarray

        if isinstance(value, np.ndarray):
            array_value = value
        elif hasattr(value, "to_numpy") and callable(getattr(value, "to_numpy")):
            array_value = np.asarray(value.to_numpy())
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise RetrievalEvaluatorInputError(f"{name} cannot be empty.")
            first_item: Any = value[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                array_value = np.asarray([np.asarray(item) for item in value], dtype=np.float64)
            else:
                array_value = np.asarray(value)
        else:
            try:
                array_value = np.asarray(value)
            except Exception as exc:  # noqa: BLE001
                raise RetrievalEvaluatorInputError(
                    f"{name} cannot be converted to numpy array: {exc}"
                ) from exc

        # Common path: object array of embedding vectors.
        if array_value.dtype == object and array_value.ndim == 1:
            if array_value.size == 0:
                raise RetrievalEvaluatorInputError(f"{name} cannot be empty.")
            try:
                stacked: np.ndarray = np.vstack(
                    [np.asarray(item, dtype=np.float64).reshape(-1) for item in array_value.tolist()]
                )
            except Exception as exc:  # noqa: BLE001
                raise RetrievalEvaluatorInputError(
                    f"{name} object payload cannot be stacked into 2D matrix: {exc}"
                ) from exc
            array_value = stacked

        if array_value.ndim == 1:
            array_value = array_value.reshape(1, -1)

        if array_value.ndim != 2:
            raise RetrievalEvaluatorInputError(
                f"{name} must be rank-2 matrix [N,D], got shape={tuple(array_value.shape)}."
            )

        if int(array_value.shape[0]) <= 0:
            raise RetrievalEvaluatorInputError(f"{name} must contain at least one sample.")
        if int(array_value.shape[1]) <= 0:
            raise RetrievalEvaluatorInputError(f"{name} must contain at least one feature.")

        embedding_matrix: np.ndarray = np.asarray(array_value, dtype=np.float64)
        if self._validate_numerics and not np.isfinite(embedding_matrix).all():
            raise RetrievalEvaluatorInputError(f"{name} contains NaN/Inf values.")

        return embedding_matrix

    def _coerce_labels(self, value: object, name: str) -> np.ndarray:
        try:
            label_array: np.ndarray = np.asarray(value)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalEvaluatorInputError(
                f"{name} cannot be converted to numpy array: {exc}"
            ) from exc

        if label_array.ndim == 0:
            label_array = label_array.reshape(1)
        if label_array.ndim > 1:
            label_array = label_array.reshape(-1)

        if int(label_array.shape[0]) <= 0:
            raise RetrievalEvaluatorInputError(f"{name} cannot be empty.")

        # Keep original label semantics but reject non-finite numeric labels.
        if np.issubdtype(label_array.dtype, np.number):
            numeric_labels: np.ndarray = np.asarray(label_array, dtype=np.float64)
            if self._validate_numerics and not np.isfinite(numeric_labels).all():
                raise RetrievalEvaluatorInputError(f"{name} contains NaN/Inf values.")
            if np.all(np.isclose(numeric_labels, np.round(numeric_labels))):
                label_array = np.asarray(np.round(numeric_labels), dtype=np.int64)
            else:
                label_array = numeric_labels

        return label_array

    def _coerce_retrieved_labels(self, value: object, name: str) -> np.ndarray:
        raw_value: Any = value
        if isinstance(value, Mapping):
            if "retrieved_labels" not in value:
                raise RetrievalEvaluatorInputError(
                    f"{name} mapping must contain key 'retrieved_labels'."
                )
            raw_value = value["retrieved_labels"]

        try:
            label_matrix: np.ndarray = np.asarray(raw_value)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalEvaluatorInputError(
                f"{name} cannot be converted to numpy array: {exc}"
            ) from exc

        if label_matrix.ndim != 2:
            raise RetrievalEvaluatorInputError(
                f"{name} must be rank-2 [N_q, K], got shape={tuple(label_matrix.shape)}."
            )

        if int(label_matrix.shape[0]) <= 0:
            raise RetrievalEvaluatorInputError(f"{name} cannot be empty.")
        if int(label_matrix.shape[1]) <= 0:
            raise RetrievalEvaluatorInputError(f"{name} must contain at least one retrieved rank.")

        return label_matrix

    def _validate_metric(self, metric: str) -> str:
        metric_str: str = "" if metric is None else str(metric).strip().lower()
        if metric_str == "":
            metric_str = DEFAULT_METRIC
        if metric_str not in _ALLOWED_METRICS:
            raise RetrievalEvaluatorConfigError(
                f"Unsupported retrieval metric={metric!r}. Allowed: {_ALLOWED_METRICS}."
            )
        return metric_str

    def _validate_top_k(self, top_k: Sequence[int]) -> Tuple[int, ...]:
        if not isinstance(top_k, Sequence):
            raise RetrievalEvaluatorConfigError(
                f"top_k must be sequence[int], got {type(top_k).__name__}."
            )

        normalized: List[int] = []
        for index, value in enumerate(top_k):
            if isinstance(value, bool):
                raise RetrievalEvaluatorConfigError(f"top_k[{index}] must be int, got bool.")
            try:
                k_value: int = int(value)
            except Exception as exc:  # noqa: BLE001
                raise RetrievalEvaluatorConfigError(
                    f"top_k[{index}] must be int, got {value!r}."
                ) from exc
            if k_value <= 0:
                raise RetrievalEvaluatorConfigError(
                    f"top_k[{index}] must be > 0, got {k_value}."
                )
            normalized.append(k_value)

        if len(normalized) == 0:
            raise RetrievalEvaluatorConfigError("top_k cannot be empty.")

        deduplicated_sorted: Tuple[int, ...] = tuple(sorted(set(normalized)))
        return deduplicated_sorted

    def _validate_k(self, k: int) -> int:
        if isinstance(k, bool):
            raise RetrievalEvaluatorInputError("k must be int, got bool.")
        try:
            k_int: int = int(k)
        except Exception as exc:  # noqa: BLE001
            raise RetrievalEvaluatorInputError(f"k must be int, got {k!r}.") from exc

        if k_int <= 0:
            raise RetrievalEvaluatorInputError(f"k must be > 0, got {k_int}.")

        return k_int


__all__ = [
    "RetrievalEvaluatorError",
    "RetrievalEvaluatorConfigError",
    "RetrievalEvaluatorInputError",
    "RetrievalEvaluatorNotReadyError",
    "RetrievalEvaluator",
]
