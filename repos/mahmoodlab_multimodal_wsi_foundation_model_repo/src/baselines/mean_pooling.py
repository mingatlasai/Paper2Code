"""Mean-pooling baseline for TITAN reproduction.

This module provides the strong unsupervised baseline used across downstream
classification, retrieval, few-shot, and survival evaluations.

Paper/config-locked contracts:
- Patch size: 512 px
- Magnification: 20x
- Patch feature dimension: 768

The baseline is parameter-free and deterministic:
- No fit/train stage
- Mask-aware arithmetic mean across valid patch features
- Strict input validation with fail-fast error handling
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import h5py
except Exception:  # pragma: no cover - optional dependency at import-time
    h5py = None  # type: ignore[assignment]

import numpy as np


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DEFAULT_EPS: float = 1.0e-12
_DEFAULT_H5_FEATURES_KEY: str = "features"
_DEFAULT_H5_MASK_KEY: str = "valid_mask"
_DEFAULT_H5_COORDS_KEY: str = "coords"
_DEFAULT_CHUNK_SIZE: int = 16_384

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


class MeanPoolingError(RuntimeError):
    """Base exception for mean-pooling baseline failures."""


class MeanPoolingSchemaError(MeanPoolingError):
    """Raised when input schema/shape contracts are violated."""


@dataclass(frozen=True)
class SlideEmbedding:
    """Container for one slide embedding result.

    Attributes:
        slide_id: Slide identifier.
        embedding: Mean-pooled embedding with shape [768].
        valid_count: Number of valid patches used.
        total_count: Number of total patches seen.
    """

    slide_id: str
    embedding: np.ndarray
    valid_count: int
    total_count: int


@dataclass(frozen=True)
class MeanPoolingStats:
    """Aggregation diagnostics for one pooling call."""

    valid_count: int
    total_count: int


class MeanPoolingBaseline:
    """Deterministic, mask-aware mean-pooling baseline."""

    def __init__(
        self,
        feature_dim: int = _FEATURE_DIM,
        patch_size_px: int = _PATCH_SIZE_PX,
        magnification: str = _MAGNIFICATION,
        eps: float = _DEFAULT_EPS,
        strict: bool = True,
    ) -> None:
        """Initialize baseline with config-locked defaults.

        Args:
            feature_dim: Feature dimension. Must be 768.
            patch_size_px: Patch size in pixels. Must be 512.
            magnification: Magnification string. Must be "20x".
            eps: Numerical epsilon for guarded divisions.
            strict: If True, enforce exact config-locked invariants.
        """
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer.")
        if isinstance(patch_size_px, bool) or not isinstance(patch_size_px, int):
            raise TypeError("patch_size_px must be an integer.")
        if not isinstance(magnification, str) or not magnification.strip():
            raise TypeError("magnification must be a non-empty string.")
        if not isinstance(eps, (int, float)):
            raise TypeError("eps must be numeric.")
        if float(eps) <= 0.0:
            raise ValueError("eps must be > 0.")

        self.feature_dim: int = int(feature_dim)
        self.patch_size_px: int = int(patch_size_px)
        self.magnification: str = str(magnification).strip()
        self.eps: float = float(eps)
        self.strict: bool = bool(strict)

        if self.strict:
            if self.feature_dim != _FEATURE_DIM:
                raise ValueError(f"feature_dim must be {_FEATURE_DIM}, got {self.feature_dim}.")
            if self.patch_size_px != _PATCH_SIZE_PX:
                raise ValueError(
                    f"patch_size_px must be {_PATCH_SIZE_PX}, got {self.patch_size_px}."
                )
            if self.magnification != _MAGNIFICATION:
                raise ValueError(
                    f"magnification must be '{_MAGNIFICATION}', got '{self.magnification}'."
                )

        # Provenance constants used by project-wide contracts.
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

    def fit(self, _: Optional[Any] = None) -> "MeanPoolingBaseline":
        """No-op for API compatibility.

        Returns:
            Self.
        """
        return self

    def embed_slide(
        self,
        features: ArrayLike,
        valid_mask: Optional[ArrayLike] = None,
    ) -> np.ndarray:
        """Compute one mean-pooled slide embedding.

        Args:
            features: Patch features with shape [N, 768].
            valid_mask: Optional patch-validity mask of length N.

        Returns:
            Mean-pooled embedding with shape [768].
        """
        embedding, _ = self.embed_slide_with_stats(features=features, valid_mask=valid_mask)
        return embedding

    def embed_slide_with_stats(
        self,
        features: ArrayLike,
        valid_mask: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, MeanPoolingStats]:
        """Compute one mean-pooled slide embedding plus diagnostics."""
        features_np: np.ndarray = self._validate_features(features=features)
        mask_np: np.ndarray = self._resolve_valid_mask(
            valid_mask=valid_mask,
            n_rows=int(features_np.shape[0]),
        )

        valid_count: int = int(np.sum(mask_np.astype(np.int64)))
        if valid_count <= 0:
            raise MeanPoolingSchemaError("No valid patches remain after masking.")

        valid_features: np.ndarray = features_np[mask_np]
        embedding: np.ndarray = np.mean(valid_features, axis=0, dtype=np.float64).astype(
            np.float32,
            copy=False,
        )

        if embedding.ndim != 1 or int(embedding.shape[0]) != self.feature_dim:
            raise MeanPoolingSchemaError(
                "Mean-pooled embedding shape mismatch: "
                f"expected ({self.feature_dim},), got {tuple(embedding.shape)}."
            )
        if not np.isfinite(embedding).all():
            raise MeanPoolingSchemaError("Mean-pooled embedding contains NaN/Inf values.")

        stats: MeanPoolingStats = MeanPoolingStats(
            valid_count=valid_count,
            total_count=int(features_np.shape[0]),
        )
        return embedding, stats

    def embed_slides(
        self,
        features_list: Sequence[ArrayLike],
        valid_masks: Optional[Sequence[Optional[ArrayLike]]] = None,
        slide_ids: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """Compute mean-pooled embeddings for multiple slides.

        Args:
            features_list: Sequence of per-slide feature matrices [Ni, 768].
            valid_masks: Optional sequence of per-slide masks.
            slide_ids: Optional sequence of slide IDs for consistency checks.

        Returns:
            Embedding matrix with shape [num_slides, 768] in input order.
        """
        if not isinstance(features_list, Sequence) or isinstance(features_list, (str, bytes)):
            raise TypeError("features_list must be a sequence of per-slide feature matrices.")
        num_slides: int = int(len(features_list))
        if num_slides <= 0:
            raise MeanPoolingSchemaError("features_list cannot be empty.")

        masks_seq: List[Optional[ArrayLike]]
        if valid_masks is None:
            masks_seq = [None] * num_slides
        else:
            if not isinstance(valid_masks, Sequence) or isinstance(valid_masks, (str, bytes)):
                raise TypeError("valid_masks must be a sequence when provided.")
            if int(len(valid_masks)) != num_slides:
                raise MeanPoolingSchemaError(
                    "valid_masks length mismatch with features_list: "
                    f"{len(valid_masks)} vs {num_slides}."
                )
            masks_seq = list(valid_masks)

        if slide_ids is not None:
            if not isinstance(slide_ids, Sequence) or isinstance(slide_ids, (str, bytes)):
                raise TypeError("slide_ids must be a sequence when provided.")
            if int(len(slide_ids)) != num_slides:
                raise MeanPoolingSchemaError(
                    "slide_ids length mismatch with features_list: "
                    f"{len(slide_ids)} vs {num_slides}."
                )

        out: np.ndarray = np.zeros((num_slides, self.feature_dim), dtype=np.float32)
        for index in range(num_slides):
            embedding: np.ndarray = self.embed_slide(
                features=features_list[index],
                valid_mask=masks_seq[index],
            )
            out[index] = embedding

        if not np.isfinite(out).all():
            raise MeanPoolingSchemaError("Batch embeddings contain NaN/Inf values.")
        return out

    def embed_h5(
        self,
        h5_path: Union[str, Path],
        features_key: str = _DEFAULT_H5_FEATURES_KEY,
        mask_key: Optional[str] = _DEFAULT_H5_MASK_KEY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> np.ndarray:
        """Compute one slide embedding from an HDF5 features artifact.

        This supports streaming accumulation to avoid loading all features at once.

        Args:
            h5_path: Path to HDF5 file containing at least `features` [N, 768].
            features_key: Dataset key for features.
            mask_key: Optional dataset key for validity mask.
            chunk_size: Number of rows per streamed chunk.

        Returns:
            Mean-pooled embedding with shape [768].
        """
        embedding, _ = self.embed_h5_with_stats(
            h5_path=h5_path,
            features_key=features_key,
            mask_key=mask_key,
            chunk_size=chunk_size,
        )
        return embedding

    def embed_h5_with_stats(
        self,
        h5_path: Union[str, Path],
        features_key: str = _DEFAULT_H5_FEATURES_KEY,
        mask_key: Optional[str] = _DEFAULT_H5_MASK_KEY,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> Tuple[np.ndarray, MeanPoolingStats]:
        """Compute one slide embedding from HDF5 with diagnostics."""
        if h5py is None:
            raise MeanPoolingError(
                "h5py is required for embed_h5/embed_h5_with_stats but is not available."
            )
        if not isinstance(features_key, str) or not features_key.strip():
            raise TypeError("features_key must be a non-empty string.")
        if mask_key is not None and (not isinstance(mask_key, str) or not mask_key.strip()):
            raise TypeError("mask_key must be None or a non-empty string.")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be an integer.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")

        resolved_path: Path = Path(h5_path).expanduser().resolve()
        if not resolved_path.exists():
            raise MeanPoolingSchemaError(f"HDF5 file does not exist: {resolved_path}")

        running_sum: np.ndarray = np.zeros((self.feature_dim,), dtype=np.float64)
        running_valid: int = 0
        total_count: int = 0

        with h5py.File(str(resolved_path), "r") as handle:
            if features_key not in handle:
                raise MeanPoolingSchemaError(
                    f"Missing features dataset '{features_key}' in {resolved_path}."
                )
            features_ds: Any = handle[features_key]
            if getattr(features_ds, "ndim", None) != 2:
                raise MeanPoolingSchemaError(
                    f"{features_key} must be rank-2 [N,D], got ndim={getattr(features_ds, 'ndim', None)}."
                )
            n_rows: int = int(features_ds.shape[0])
            n_dim: int = int(features_ds.shape[1])
            if n_rows <= 0:
                raise MeanPoolingSchemaError("features dataset is empty.")
            if n_dim != self.feature_dim:
                raise MeanPoolingSchemaError(
                    f"Feature dim must be {self.feature_dim}, got {n_dim}."
                )

            mask_ds: Optional[Any] = None
            if mask_key is not None and mask_key in handle:
                mask_ds = handle[mask_key]
                if int(mask_ds.shape[0]) != n_rows:
                    raise MeanPoolingSchemaError(
                        "Mask length mismatch with features rows: "
                        f"{int(mask_ds.shape[0])} vs {n_rows}."
                    )

            for start in range(0, n_rows, int(chunk_size)):
                end: int = min(start + int(chunk_size), n_rows)
                feat_chunk: np.ndarray = np.asarray(features_ds[start:end], dtype=np.float32)
                feat_chunk = self._validate_features(features=feat_chunk)

                if mask_ds is not None:
                    mask_chunk_raw: np.ndarray = np.asarray(mask_ds[start:end])
                    mask_chunk: np.ndarray = self._resolve_valid_mask(
                        valid_mask=mask_chunk_raw,
                        n_rows=int(feat_chunk.shape[0]),
                    )
                else:
                    mask_chunk = np.ones((int(feat_chunk.shape[0]),), dtype=np.bool_)

                valid_chunk_count: int = int(np.sum(mask_chunk.astype(np.int64)))
                if valid_chunk_count > 0:
                    running_sum += np.sum(feat_chunk[mask_chunk], axis=0, dtype=np.float64)
                    running_valid += valid_chunk_count
                total_count += int(feat_chunk.shape[0])

        if running_valid <= 0:
            raise MeanPoolingSchemaError("No valid patches were available for mean pooling.")

        embedding: np.ndarray = (running_sum / max(float(running_valid), self.eps)).astype(
            np.float32,
            copy=False,
        )
        if embedding.ndim != 1 or int(embedding.shape[0]) != self.feature_dim:
            raise MeanPoolingSchemaError(
                f"Streaming embedding shape mismatch: got {tuple(embedding.shape)}."
            )
        if not np.isfinite(embedding).all():
            raise MeanPoolingSchemaError("Streaming mean-pooled embedding contains NaN/Inf.")

        stats: MeanPoolingStats = MeanPoolingStats(
            valid_count=int(running_valid),
            total_count=int(total_count),
        )
        return embedding, stats

    def build_slide_records(
        self,
        slide_to_features: Mapping[str, ArrayLike],
        slide_to_mask: Optional[Mapping[str, ArrayLike]] = None,
    ) -> List[SlideEmbedding]:
        """Compute typed slide embedding records from mapping inputs."""
        if not isinstance(slide_to_features, Mapping):
            raise TypeError("slide_to_features must be a mapping of slide_id -> features.")
        if len(slide_to_features) == 0:
            raise MeanPoolingSchemaError("slide_to_features cannot be empty.")

        mask_map: Mapping[str, ArrayLike]
        if slide_to_mask is None:
            mask_map = {}
        else:
            if not isinstance(slide_to_mask, Mapping):
                raise TypeError("slide_to_mask must be a mapping when provided.")
            mask_map = slide_to_mask

        records: List[SlideEmbedding] = []
        for slide_id in slide_to_features:
            feature_obj: ArrayLike = slide_to_features[slide_id]
            mask_obj: Optional[ArrayLike] = mask_map.get(slide_id)
            embedding, stats = self.embed_slide_with_stats(
                features=feature_obj,
                valid_mask=mask_obj,
            )
            records.append(
                SlideEmbedding(
                    slide_id=str(slide_id),
                    embedding=embedding,
                    valid_count=int(stats.valid_count),
                    total_count=int(stats.total_count),
                )
            )
        return records

    def aggregate_patients(
        self,
        slide_embeddings: np.ndarray,
        patient_ids: Sequence[str],
        slide_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Aggregate slide embeddings into patient-level mean embeddings.

        Args:
            slide_embeddings: Slide embedding matrix [N, 768].
            patient_ids: Patient identifier for each slide.
            slide_ids: Optional slide ids aligned to rows.

        Returns:
            Dict with keys:
            - patient_ids: ordered list of unique patient IDs
            - embeddings: patient embedding matrix [P, 768]
            - counts: number of slides per patient [P]
            - groups: patient -> list of slide IDs (if provided)
        """
        x: np.ndarray = self._validate_embedding_matrix(slide_embeddings)

        if not isinstance(patient_ids, Sequence) or isinstance(patient_ids, (str, bytes)):
            raise TypeError("patient_ids must be a sequence of strings.")
        if int(len(patient_ids)) != int(x.shape[0]):
            raise MeanPoolingSchemaError(
                "patient_ids length must match number of slide embeddings: "
                f"{len(patient_ids)} vs {int(x.shape[0])}."
            )

        slide_id_list: Optional[List[str]] = None
        if slide_ids is not None:
            if not isinstance(slide_ids, Sequence) or isinstance(slide_ids, (str, bytes)):
                raise TypeError("slide_ids must be a sequence when provided.")
            if int(len(slide_ids)) != int(x.shape[0]):
                raise MeanPoolingSchemaError(
                    "slide_ids length must match number of slide embeddings: "
                    f"{len(slide_ids)} vs {int(x.shape[0])}."
                )
            slide_id_list = [str(item) for item in slide_ids]

        patient_order: List[str] = []
        patient_sum: Dict[str, np.ndarray] = {}
        patient_count: Dict[str, int] = {}
        patient_groups: Dict[str, List[str]] = {}

        for index, patient_id_obj in enumerate(patient_ids):
            patient_id: str = str(patient_id_obj)
            if not patient_id:
                raise MeanPoolingSchemaError(f"Empty patient_id at row index {index}.")

            if patient_id not in patient_sum:
                patient_order.append(patient_id)
                patient_sum[patient_id] = np.zeros((self.feature_dim,), dtype=np.float64)
                patient_count[patient_id] = 0
                patient_groups[patient_id] = []

            patient_sum[patient_id] += x[index].astype(np.float64, copy=False)
            patient_count[patient_id] += 1

            if slide_id_list is not None:
                patient_groups[patient_id].append(slide_id_list[index])

        p_count: int = int(len(patient_order))
        out_emb: np.ndarray = np.zeros((p_count, self.feature_dim), dtype=np.float32)
        out_counts: np.ndarray = np.zeros((p_count,), dtype=np.int64)
        for p_index, patient_id in enumerate(patient_order):
            count_value: int = int(patient_count[patient_id])
            if count_value <= 0:
                raise MeanPoolingSchemaError(
                    f"Patient '{patient_id}' has zero slides during aggregation."
                )
            out_emb[p_index] = (patient_sum[patient_id] / float(count_value)).astype(
                np.float32,
                copy=False,
            )
            out_counts[p_index] = count_value

        if not np.isfinite(out_emb).all():
            raise MeanPoolingSchemaError("Patient embeddings contain NaN/Inf values.")

        result: Dict[str, Any] = {
            "patient_ids": patient_order,
            "embeddings": out_emb,
            "counts": out_counts,
        }
        if slide_id_list is not None:
            result["groups"] = patient_groups
        return result

    def _validate_features(self, features: Any) -> np.ndarray:
        """Validate per-slide patch features with shape [N, 768]."""
        features_np: np.ndarray = self._to_numpy_float32(features)
        if features_np.ndim != 2:
            raise MeanPoolingSchemaError(
                f"features must be rank-2 [N,D], got {tuple(features_np.shape)}."
            )
        if int(features_np.shape[0]) <= 0:
            raise MeanPoolingSchemaError("features must contain at least one row.")
        if int(features_np.shape[1]) != self.feature_dim:
            raise MeanPoolingSchemaError(
                f"features second dimension must be {self.feature_dim}, got {int(features_np.shape[1])}."
            )
        if not np.isfinite(features_np).all():
            raise MeanPoolingSchemaError("features contain NaN/Inf values.")
        return features_np

    def _validate_embedding_matrix(self, x: Any) -> np.ndarray:
        """Validate an embedding matrix [N, 768]."""
        matrix: np.ndarray = self._to_numpy_float32(x)
        if matrix.ndim != 2:
            raise MeanPoolingSchemaError(
                f"slide_embeddings must be rank-2 [N,D], got {tuple(matrix.shape)}."
            )
        if int(matrix.shape[0]) <= 0:
            raise MeanPoolingSchemaError("slide_embeddings cannot be empty.")
        if int(matrix.shape[1]) != self.feature_dim:
            raise MeanPoolingSchemaError(
                "slide_embeddings second dimension mismatch: "
                f"expected {self.feature_dim}, got {int(matrix.shape[1])}."
            )
        if not np.isfinite(matrix).all():
            raise MeanPoolingSchemaError("slide_embeddings contain NaN/Inf values.")
        return matrix

    def _resolve_valid_mask(self, valid_mask: Optional[Any], n_rows: int) -> np.ndarray:
        """Resolve optional mask into bool vector [N]."""
        if n_rows <= 0:
            raise MeanPoolingSchemaError("n_rows must be > 0 when resolving valid mask.")

        if valid_mask is None:
            return np.ones((n_rows,), dtype=np.bool_)

        mask_arr: np.ndarray = np.asarray(valid_mask)
        if mask_arr.ndim == 2:
            if int(mask_arr.shape[0]) == n_rows and int(mask_arr.shape[1]) == 1:
                mask_arr = mask_arr[:, 0]
            elif int(mask_arr.shape[0]) == 1 and int(mask_arr.shape[1]) == n_rows:
                mask_arr = mask_arr[0, :]
            else:
                raise MeanPoolingSchemaError(
                    f"valid_mask rank-2 shape not supported: {tuple(mask_arr.shape)} for n_rows={n_rows}."
                )

        if mask_arr.ndim != 1:
            raise MeanPoolingSchemaError(
                f"valid_mask must be rank-1 [N], got shape={tuple(mask_arr.shape)}."
            )
        if int(mask_arr.shape[0]) != n_rows:
            raise MeanPoolingSchemaError(
                f"valid_mask length mismatch: {int(mask_arr.shape[0])} vs expected {n_rows}."
            )

        if np.issubdtype(mask_arr.dtype, np.bool_):
            mask_bool: np.ndarray = mask_arr.astype(np.bool_, copy=False)
        elif np.issubdtype(mask_arr.dtype, np.integer):
            unique_values: np.ndarray = np.unique(mask_arr.astype(np.int64, copy=False))
            if not set(int(v) for v in unique_values.tolist()).issubset({0, 1}):
                raise MeanPoolingSchemaError(
                    "Integer valid_mask values must be binary {0,1}, got "
                    f"{unique_values.tolist()}."
                )
            mask_bool = mask_arr.astype(np.bool_, copy=False)
        elif np.issubdtype(mask_arr.dtype, np.floating):
            if not np.all(np.isfinite(mask_arr)):
                raise MeanPoolingSchemaError("valid_mask contains NaN/Inf values.")
            if not np.all(np.equal(mask_arr, np.floor(mask_arr))):
                raise MeanPoolingSchemaError(
                    "Floating valid_mask values must be integer-valued 0/1."
                )
            unique_values = np.unique(mask_arr.astype(np.int64, copy=False))
            if not set(int(v) for v in unique_values.tolist()).issubset({0, 1}):
                raise MeanPoolingSchemaError(
                    "Floating valid_mask values must be binary {0,1}, got "
                    f"{unique_values.tolist()}."
                )
            mask_bool = mask_arr.astype(np.bool_, copy=False)
        else:
            raise MeanPoolingSchemaError(
                f"Unsupported valid_mask dtype: {mask_arr.dtype}."
            )

        return mask_bool

    @staticmethod
    def _to_numpy_float32(value: Any) -> np.ndarray:
        """Convert input array-like to finite float32 NumPy array."""
        # Optional torch compatibility without hard dependency.
        if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
            try:
                value = value.detach().cpu().numpy()
            except Exception as exc:  # pragma: no cover - defensive branch
                raise MeanPoolingSchemaError(
                    "Failed converting tensor-like input to numpy."
                ) from exc

        array: np.ndarray = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise MeanPoolingSchemaError("Input contains NaN/Inf values.")
        return array


class Baseline(MeanPoolingBaseline):
    """Design-compat alias for baseline creation."""


def mean_pool_slide(features: ArrayLike, valid_mask: Optional[ArrayLike] = None) -> np.ndarray:
    """Functional API for one-slide mean pooling."""
    baseline: MeanPoolingBaseline = MeanPoolingBaseline()
    return baseline.embed_slide(features=features, valid_mask=valid_mask)


def mean_pool_slides(
    features_list: Sequence[ArrayLike],
    valid_masks: Optional[Sequence[Optional[ArrayLike]]] = None,
) -> np.ndarray:
    """Functional API for batch mean pooling."""
    baseline = MeanPoolingBaseline()
    return baseline.embed_slides(features_list=features_list, valid_masks=valid_masks)


def aggregate_patient_embeddings(
    slide_embeddings: np.ndarray,
    patient_ids: Sequence[str],
    slide_ids: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Functional API for patient-level mean aggregation."""
    baseline = MeanPoolingBaseline()
    return baseline.aggregate_patients(
        slide_embeddings=slide_embeddings,
        patient_ids=patient_ids,
        slide_ids=slide_ids,
    )


def run_mean_pooling(features: ArrayLike, valid_mask: Optional[ArrayLike] = None) -> Dict[str, Any]:
    """Design-friendly helper returning embedding and minimal diagnostics."""
    baseline: MeanPoolingBaseline = MeanPoolingBaseline()
    embedding, stats = baseline.embed_slide_with_stats(features=features, valid_mask=valid_mask)
    return {
        "embedding": embedding,
        "valid_count": int(stats.valid_count),
        "total_count": int(stats.total_count),
        "feature_dim": int(baseline.feature_dim),
    }


__all__ = [
    "MeanPoolingError",
    "MeanPoolingSchemaError",
    "MeanPoolingStats",
    "SlideEmbedding",
    "MeanPoolingBaseline",
    "Baseline",
    "mean_pool_slide",
    "mean_pool_slides",
    "aggregate_patient_embeddings",
    "run_mean_pooling",
]
