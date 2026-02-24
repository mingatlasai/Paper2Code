"""Persistent patch feature storage for THREADS reproduction.

This module implements the design-locked ``FeatureStore`` interface:
- ``FeatureStore.__init__(root_dir: str, fmt: str) -> None``
- ``FeatureStore.write_patch_features(sample_id: str, features: object, coords: object) -> str``
- ``FeatureStore.read_patch_features(sample_id: str) -> tuple[object, object]``
- ``FeatureStore.exists(sample_id: str) -> bool``
- ``FeatureStore.delete(sample_id: str) -> None``

Paper/config-aligned defaults:
- target magnification: 20x
- patch size: 512
- patch stride: 512 (no overlap)
- required feature store schema keys:
  ``sample_id``, ``coords``, ``features``, ``encoder_name``, ``precision``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import hashlib
import re

import numpy as np
import pandas as pd

from src.utils.io import (
    FEATURE_STORE_REQUIRED_KEYS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_TARGET_MAGNIFICATION,
    ensure_dir,
    read_feature_store_sample,
    read_h5_attrs,
    read_json,
    read_parquet,
    validate_feature_pair,
    write_feature_store_sample,
    write_json,
    write_parquet,
)


DEFAULT_FORMAT: str = "hdf5"
DEFAULT_ENCODER_NAME: str = "CONCHV1.5"
DEFAULT_PRECISION: str = "fp32"
DEFAULT_TARGET_MAGNIFICATION_FOR_FEATURES: str = DEFAULT_TARGET_MAGNIFICATION
DEFAULT_PATCH_SIZE_FOR_FEATURES: int = DEFAULT_PATCH_SIZE
DEFAULT_PARQUET_FEATURE_COLUMN_PREFIX: str = "f_"
DEFAULT_PARQUET_COORD_X_COLUMN: str = "x"
DEFAULT_PARQUET_COORD_Y_COLUMN: str = "y"
DEFAULT_PARQUET_SAMPLE_ID_COLUMN: str = "sample_id"
DEFAULT_PARQUET_PATCH_INDEX_COLUMN: str = "patch_index"
DEFAULT_HASH_PREFIX_LENGTH: int = 2

_SUPPORTED_FORMATS: Tuple[str, ...] = ("hdf5", "h5", "parquet")


class FeatureStoreError(Exception):
    """Base exception for feature-store operations."""


class FeatureStoreConfigError(FeatureStoreError):
    """Raised when feature-store configuration is invalid."""


class FeatureStoreWriteError(FeatureStoreError):
    """Raised when feature writing fails."""


class FeatureStoreReadError(FeatureStoreError):
    """Raised when feature reading fails."""


class FeatureStoreIntegrityError(FeatureStoreError):
    """Raised when stored artifacts violate schema/integrity constraints."""


@dataclass(frozen=True)
class _StorePathBundle:
    """Resolved file paths for one sample artifact."""

    feature_path: Path
    meta_path: Optional[Path]


class FeatureStore:
    """Persistent per-sample patch feature storage.

    The store is deterministic and sample-id keyed. It supports two formats:
    - ``hdf5``/``h5``: dense arrays in a single HDF5 file.
    - ``parquet``: row-wise table with coords + feature columns plus JSON sidecar metadata.
    """

    def __init__(self, root_dir: str, fmt: str) -> None:
        """Initialize feature store.

        Args:
            root_dir: Root artifact directory.
            fmt: Storage format key in {"hdf5", "h5", "parquet"}.

        Raises:
            FeatureStoreConfigError: If configuration is invalid.
        """
        normalized_root_dir: str = str(root_dir).strip() if root_dir is not None else ""
        if not normalized_root_dir:
            raise FeatureStoreConfigError("root_dir must be a non-empty path string.")

        normalized_format: str = self._normalize_format(fmt)
        resolved_root_dir: Path = ensure_dir(normalized_root_dir)

        self._root_dir: Path = resolved_root_dir
        self._fmt: str = normalized_format
        self._encoder_name: str = DEFAULT_ENCODER_NAME
        self._precision: str = DEFAULT_PRECISION

    def write_patch_features(self, sample_id: str, features: object, coords: object) -> str:
        """Persist patch features and aligned coordinates for one sample.

        Args:
            sample_id: Stable manifest sample identifier.
            features: Patch feature array-like of shape [N, D].
            coords: Coordinate array-like of shape [N, 2].

        Returns:
            String URI/path to persisted artifact.

        Raises:
            FeatureStoreWriteError: If validation or persistence fails.
        """
        validated_sample_id: str = self._validate_sample_id(sample_id)

        feature_array: np.ndarray = np.asarray(features)
        coord_array: np.ndarray = np.asarray(coords)

        try:
            validate_feature_pair(
                sample_id=validated_sample_id,
                features=feature_array,
                coords=coord_array,
                expected_patch_size=DEFAULT_PATCH_SIZE_FOR_FEATURES,
            )
        except Exception as exc:  # noqa: BLE001
            raise FeatureStoreWriteError(
                f"Invalid features/coords for sample_id={validated_sample_id}: {exc}"
            ) from exc

        # Normalize dtypes for stable storage/retrieval.
        feature_array = np.asarray(feature_array, dtype=np.float32)
        coord_array = np.asarray(coord_array, dtype=np.int64)

        bundle: _StorePathBundle = self._resolve_paths(validated_sample_id)

        try:
            if self._fmt in {"hdf5", "h5"}:
                write_feature_store_sample(
                    path=bundle.feature_path,
                    sample_id=validated_sample_id,
                    features=feature_array,
                    coords=coord_array,
                    encoder_name=self._encoder_name,
                    precision=self._precision,
                    patch_size=DEFAULT_PATCH_SIZE_FOR_FEATURES,
                    target_magnification=DEFAULT_TARGET_MAGNIFICATION_FOR_FEATURES,
                )
            else:
                self._write_parquet_sample(
                    sample_id=validated_sample_id,
                    feature_path=bundle.feature_path,
                    meta_path=bundle.meta_path,
                    features=feature_array,
                    coords=coord_array,
                )
        except Exception as exc:  # noqa: BLE001
            raise FeatureStoreWriteError(
                f"Failed writing features for sample_id={validated_sample_id}: {exc}"
            ) from exc

        return str(bundle.feature_path)

    def read_patch_features(self, sample_id: str) -> tuple[object, object]:
        """Read persisted patch features and coordinates.

        Args:
            sample_id: Stable manifest sample identifier.

        Returns:
            Tuple ``(features, coords)`` where both are numpy arrays.

        Raises:
            FeatureStoreReadError: If artifact is missing/unreadable.
            FeatureStoreIntegrityError: If artifact content is inconsistent.
        """
        validated_sample_id: str = self._validate_sample_id(sample_id)
        bundle: _StorePathBundle = self._resolve_paths(validated_sample_id)

        if not bundle.feature_path.exists():
            raise FeatureStoreReadError(
                f"Feature artifact not found for sample_id={validated_sample_id}: {bundle.feature_path}"
            )

        if self._fmt in {"hdf5", "h5"}:
            try:
                features, coords, attrs = read_feature_store_sample(bundle.feature_path)
            except Exception as exc:  # noqa: BLE001
                raise FeatureStoreReadError(
                    f"Failed reading HDF5 features for sample_id={validated_sample_id}: {exc}"
                ) from exc

            stored_sample_id: str = str(attrs.get("sample_id", "")).strip()
            if stored_sample_id != validated_sample_id:
                raise FeatureStoreIntegrityError(
                    "Sample identity mismatch in HDF5 feature artifact. "
                    f"requested={validated_sample_id}, stored={stored_sample_id}"
                )

            self._validate_required_metadata(attrs, sample_id=validated_sample_id)
            return features, coords

        try:
            features, coords = self._read_parquet_sample(
                sample_id=validated_sample_id,
                feature_path=bundle.feature_path,
                meta_path=bundle.meta_path,
            )
        except FeatureStoreIntegrityError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FeatureStoreReadError(
                f"Failed reading parquet features for sample_id={validated_sample_id}: {exc}"
            ) from exc

        return features, coords

    def exists(self, sample_id: str) -> bool:
        """Check whether a complete/valid artifact exists for sample_id.

        This method is integrity-aware: it verifies required metadata and minimal
        schema consistency rather than only checking path existence.

        Args:
            sample_id: Stable manifest sample identifier.

        Returns:
            True if artifact exists and passes minimal integrity checks.
        """
        try:
            validated_sample_id: str = self._validate_sample_id(sample_id)
        except FeatureStoreError:
            return False

        bundle: _StorePathBundle = self._resolve_paths(validated_sample_id)
        if not bundle.feature_path.exists() or not bundle.feature_path.is_file():
            return False

        try:
            if self._fmt in {"hdf5", "h5"}:
                attrs: Dict[str, Any] = read_h5_attrs(bundle.feature_path)
                stored_sample_id: str = str(attrs.get("sample_id", "")).strip()
                if stored_sample_id != validated_sample_id:
                    return False
                self._validate_required_metadata(attrs, sample_id=validated_sample_id)
                return True

            if bundle.meta_path is None or not bundle.meta_path.exists() or not bundle.meta_path.is_file():
                return False

            meta_payload: Dict[str, Any] = read_json(bundle.meta_path)
            self._validate_required_metadata(meta_payload, sample_id=validated_sample_id)

            parquet_df: pd.DataFrame = read_parquet(
                bundle.feature_path,
                columns=[DEFAULT_PARQUET_SAMPLE_ID_COLUMN, DEFAULT_PARQUET_PATCH_INDEX_COLUMN],
            )
            if parquet_df.empty:
                return False
            if DEFAULT_PARQUET_SAMPLE_ID_COLUMN not in parquet_df.columns:
                return False
            unique_sample_ids: List[str] = sorted(
                set(parquet_df[DEFAULT_PARQUET_SAMPLE_ID_COLUMN].astype(str).tolist())
            )
            if unique_sample_ids != [validated_sample_id]:
                return False
            return True
        except Exception:
            return False

    def delete(self, sample_id: str) -> None:
        """Delete persisted artifact for one sample.

        Args:
            sample_id: Stable manifest sample identifier.

        Notes:
            This operation only removes files mapped to the target sample_id and
            never removes directories.
        """
        validated_sample_id: str = self._validate_sample_id(sample_id)
        bundle: _StorePathBundle = self._resolve_paths(validated_sample_id)

        if bundle.feature_path.exists() and bundle.feature_path.is_file():
            bundle.feature_path.unlink(missing_ok=True)

        if bundle.meta_path is not None and bundle.meta_path.exists() and bundle.meta_path.is_file():
            bundle.meta_path.unlink(missing_ok=True)

    def _normalize_format(self, fmt: str) -> str:
        """Normalize and validate storage format."""
        normalized_format: str = str(fmt).strip().lower() if fmt is not None else ""
        if normalized_format == "":
            normalized_format = DEFAULT_FORMAT

        alias_map: Dict[str, str] = {
            "hdf5": "hdf5",
            "h5": "h5",
            "parquet": "parquet",
        }
        if normalized_format not in alias_map:
            raise FeatureStoreConfigError(
                f"Unsupported fmt={fmt!r}. Expected one of {_SUPPORTED_FORMATS}."
            )
        return alias_map[normalized_format]

    def _validate_sample_id(self, sample_id: str) -> str:
        """Validate and sanitize sample identifier."""
        normalized_sample_id: str = "" if sample_id is None else str(sample_id).strip()
        if not normalized_sample_id:
            raise FeatureStoreConfigError("sample_id must be a non-empty string.")

        # Keep deterministic path safety.
        if "/" in normalized_sample_id or "\\" in normalized_sample_id:
            raise FeatureStoreConfigError(
                f"sample_id cannot contain path separators: {normalized_sample_id!r}"
            )

        return normalized_sample_id

    def _resolve_paths(self, sample_id: str) -> _StorePathBundle:
        """Resolve deterministic feature and sidecar paths for a sample."""
        safe_name: str = self._safe_file_stem(sample_id)
        shard: str = self._shard_prefix(sample_id)
        sample_dir: Path = ensure_dir(self._root_dir / shard)

        if self._fmt in {"hdf5", "h5"}:
            suffix: str = ".h5" if self._fmt == "h5" else ".hdf5"
            feature_path: Path = sample_dir / f"{safe_name}{suffix}"
            return _StorePathBundle(feature_path=feature_path, meta_path=None)

        feature_path = sample_dir / f"{safe_name}.parquet"
        meta_path: Path = sample_dir / f"{safe_name}.meta.json"
        return _StorePathBundle(feature_path=feature_path, meta_path=meta_path)

    def _safe_file_stem(self, sample_id: str) -> str:
        """Return filesystem-safe stable file stem from sample_id."""
        compact: str = sample_id.strip()
        compact = re.sub(r"\s+", "_", compact)
        compact = re.sub(r"[^A-Za-z0-9_.-]", "_", compact)
        if compact == "":
            compact = "sample"
        return compact

    def _shard_prefix(self, sample_id: str) -> str:
        """Return deterministic shard prefix from sample_id hash."""
        digest: str = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()
        return digest[:DEFAULT_HASH_PREFIX_LENGTH]

    def _write_parquet_sample(
        self,
        sample_id: str,
        feature_path: Path,
        meta_path: Optional[Path],
        features: np.ndarray,
        coords: np.ndarray,
    ) -> None:
        """Write one sample as parquet + JSON sidecar metadata."""
        if meta_path is None:
            raise FeatureStoreWriteError("Parquet backend requires meta sidecar path.")

        num_patches: int = int(features.shape[0])
        feature_dim: int = int(features.shape[1])

        data: Dict[str, Any] = {
            DEFAULT_PARQUET_SAMPLE_ID_COLUMN: np.asarray([sample_id] * num_patches, dtype=object),
            DEFAULT_PARQUET_PATCH_INDEX_COLUMN: np.arange(num_patches, dtype=np.int64),
            DEFAULT_PARQUET_COORD_X_COLUMN: np.asarray(coords[:, 0], dtype=np.int64),
            DEFAULT_PARQUET_COORD_Y_COLUMN: np.asarray(coords[:, 1], dtype=np.int64),
        }

        feature_index: int
        for feature_index in range(feature_dim):
            column_name: str = f"{DEFAULT_PARQUET_FEATURE_COLUMN_PREFIX}{feature_index:04d}"
            data[column_name] = np.asarray(features[:, feature_index], dtype=np.float32)

        frame: pd.DataFrame = pd.DataFrame(data)
        write_parquet(frame, feature_path, sort_columns=False)

        metadata: Dict[str, Any] = {
            "sample_id": sample_id,
            "encoder_name": self._encoder_name,
            "precision": self._precision,
            "patch_size": int(DEFAULT_PATCH_SIZE_FOR_FEATURES),
            "target_magnification": str(DEFAULT_TARGET_MAGNIFICATION_FOR_FEATURES),
            "feature_dim": int(feature_dim),
            "num_patches": int(num_patches),
            "schema_required_keys": list(FEATURE_STORE_REQUIRED_KEYS),
            "format": "parquet",
        }
        write_json(metadata, meta_path)

    def _read_parquet_sample(
        self,
        sample_id: str,
        feature_path: Path,
        meta_path: Optional[Path],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read one parquet sample artifact and validate integrity."""
        if meta_path is None:
            raise FeatureStoreIntegrityError("Parquet backend requires meta sidecar path.")
        if not meta_path.exists() or not meta_path.is_file():
            raise FeatureStoreIntegrityError(
                f"Missing parquet metadata sidecar for sample_id={sample_id}: {meta_path}"
            )

        metadata: Dict[str, Any] = read_json(meta_path)
        self._validate_required_metadata(metadata, sample_id=sample_id)

        frame: pd.DataFrame = read_parquet(feature_path)
        if frame.empty:
            raise FeatureStoreIntegrityError(
                f"Parquet feature artifact is empty for sample_id={sample_id}: {feature_path}"
            )

        required_columns: Tuple[str, ...] = (
            DEFAULT_PARQUET_SAMPLE_ID_COLUMN,
            DEFAULT_PARQUET_PATCH_INDEX_COLUMN,
            DEFAULT_PARQUET_COORD_X_COLUMN,
            DEFAULT_PARQUET_COORD_Y_COLUMN,
        )
        missing_columns: List[str] = [
            column_name for column_name in required_columns if column_name not in frame.columns
        ]
        if missing_columns:
            raise FeatureStoreIntegrityError(
                f"Parquet artifact missing required columns for sample_id={sample_id}: {missing_columns}"
            )

        sample_values: List[str] = sorted(
            set(frame[DEFAULT_PARQUET_SAMPLE_ID_COLUMN].astype(str).tolist())
        )
        if sample_values != [sample_id]:
            raise FeatureStoreIntegrityError(
                "Parquet sample identity mismatch. "
                f"requested={sample_id}, stored={sample_values}"
            )

        frame = frame.sort_values(DEFAULT_PARQUET_PATCH_INDEX_COLUMN, kind="mergesort").reset_index(drop=True)

        feature_columns: List[str] = [
            column_name
            for column_name in frame.columns
            if column_name.startswith(DEFAULT_PARQUET_FEATURE_COLUMN_PREFIX)
        ]
        if len(feature_columns) == 0:
            raise FeatureStoreIntegrityError(
                f"No feature columns found for sample_id={sample_id}."
            )
        feature_columns.sort(key=self._feature_column_sort_key)

        coords: np.ndarray = frame[
            [DEFAULT_PARQUET_COORD_X_COLUMN, DEFAULT_PARQUET_COORD_Y_COLUMN]
        ].to_numpy(dtype=np.int64)
        features: np.ndarray = frame[feature_columns].to_numpy(dtype=np.float32)

        try:
            validate_feature_pair(
                sample_id=sample_id,
                features=features,
                coords=coords,
                expected_patch_size=DEFAULT_PATCH_SIZE_FOR_FEATURES,
            )
        except Exception as exc:  # noqa: BLE001
            raise FeatureStoreIntegrityError(
                f"Invalid reconstructed parquet arrays for sample_id={sample_id}: {exc}"
            ) from exc

        expected_feature_dim_raw: Any = metadata.get("feature_dim", features.shape[1])
        expected_feature_dim: int = int(expected_feature_dim_raw)
        if int(features.shape[1]) != expected_feature_dim:
            raise FeatureStoreIntegrityError(
                "Feature dimension mismatch between parquet metadata and payload. "
                f"sample_id={sample_id}, metadata_dim={expected_feature_dim}, payload_dim={features.shape[1]}"
            )

        return features, coords

    def _feature_column_sort_key(self, column_name: str) -> int:
        """Sort key for feature columns f_0000, f_0001, ..."""
        suffix: str = column_name.replace(DEFAULT_PARQUET_FEATURE_COLUMN_PREFIX, "", 1)
        try:
            return int(suffix)
        except ValueError:
            return 10**12

    def _validate_required_metadata(self, metadata: Mapping[str, Any], sample_id: str) -> None:
        """Validate required schema metadata keys and values."""
        if not isinstance(metadata, Mapping):
            raise FeatureStoreIntegrityError(
                f"Metadata must be a mapping for sample_id={sample_id}."
            )

        missing_keys: List[str] = [
            key_name for key_name in FEATURE_STORE_REQUIRED_KEYS if key_name not in metadata
        ]
        if missing_keys:
            raise FeatureStoreIntegrityError(
                f"Missing required metadata keys for sample_id={sample_id}: {missing_keys}"
            )

        stored_sample_id: str = str(metadata.get("sample_id", "")).strip()
        if stored_sample_id != sample_id:
            raise FeatureStoreIntegrityError(
                "Metadata sample identity mismatch. "
                f"requested={sample_id}, stored={stored_sample_id}"
            )

        encoder_name: str = str(metadata.get("encoder_name", "")).strip()
        precision: str = str(metadata.get("precision", "")).strip()

        if encoder_name == "":
            raise FeatureStoreIntegrityError(
                f"Metadata encoder_name is empty for sample_id={sample_id}."
            )
        if precision == "":
            raise FeatureStoreIntegrityError(
                f"Metadata precision is empty for sample_id={sample_id}."
            )


__all__ = [
    "DEFAULT_FORMAT",
    "DEFAULT_ENCODER_NAME",
    "DEFAULT_PRECISION",
    "FeatureStoreError",
    "FeatureStoreConfigError",
    "FeatureStoreWriteError",
    "FeatureStoreReadError",
    "FeatureStoreIntegrityError",
    "FeatureStore",
]
