"""Patch tiling and extraction for WSI preprocessing.

This module implements the design-locked ``PatchTiler`` interface:
- ``PatchTiler.__init__(patch_size: int, stride: int, target_magnification: int) -> None``
- ``PatchTiler.tile(slide: object, tissue_mask: object) -> list[tuple[int, int]]``
- ``PatchTiler.extract(slide: object, coords: list[tuple[int, int]]) -> list[bytes]``

Paper/config alignment (from ``config.yaml``):
- ``target_magnification = 20x``
- ``patch_size = 512``
- ``patch_stride = 512``
- ``overlap = 0``

Coordinate convention:
- Public ``coords`` are level-0 top-left pixel coordinates.
- Extraction reads patches at the pyramid level closest to target magnification,
  returning raw RGB bytes of shape ``(patch_size, patch_size, 3)`` per patch.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np


DEFAULT_PATCH_SIZE: int = 512
DEFAULT_PATCH_STRIDE: int = 512
DEFAULT_TARGET_MAGNIFICATION: int = 20
DEFAULT_OVERLAP: int = 0

# "Process patches overlapping tissue" from paper text. Any positive overlap passes.
DEFAULT_MIN_TISSUE_FRACTION: float = 0.0

class PatchTilerError(Exception):
    """Base exception for patch tiling/extraction failures."""


class PatchTilerConfigError(PatchTilerError):
    """Raised when constructor arguments violate required constraints."""


class PatchTilerMetadataError(PatchTilerError):
    """Raised when slide/mask metadata is missing or inconsistent."""


class PatchTilerMaskError(PatchTilerError):
    """Raised when tissue mask payload is invalid."""


class PatchTilerReadError(PatchTilerError):
    """Raised when patch extraction fails."""


@dataclass(frozen=True)
class _SlideMeta:
    """Normalized slide metadata required by tiling and extraction."""

    backend: str
    width_level0: int
    height_level0: int
    level_count: int
    level_downsamples: Tuple[float, ...]
    level_dimensions: Tuple[Tuple[int, int], ...]
    objective_power: Optional[float]
    mpp: Optional[float]


@dataclass(frozen=True)
class _MaskMeta:
    """Normalized tissue mask metadata."""

    mask: np.ndarray
    level: int
    downsample: float


class PatchTiler:
    """Deterministic non-overlapping tissue-aware patch tiler."""

    def __init__(self, patch_size: int, stride: int, target_magnification: int) -> None:
        """Initialize tiler with strict paper/config constraints.

        Args:
            patch_size: Patch side length in pixels. Must be 512.
            stride: Grid stride in pixels. Must be 512.
            target_magnification: Target optical magnification. Must be 20.

        Raises:
            PatchTilerConfigError: If any constraint is violated.
        """
        patch_size_int: int = _coerce_positive_int(patch_size, "patch_size")
        stride_int: int = _coerce_positive_int(stride, "stride")
        target_magnification_int: int = _coerce_positive_int(
            target_magnification,
            "target_magnification",
        )

        if patch_size_int != DEFAULT_PATCH_SIZE:
            raise PatchTilerConfigError(
                f"patch_size must be {DEFAULT_PATCH_SIZE}, got {patch_size_int}."
            )
        if stride_int != DEFAULT_PATCH_STRIDE:
            raise PatchTilerConfigError(
                f"stride must be {DEFAULT_PATCH_STRIDE}, got {stride_int}."
            )
        if target_magnification_int != DEFAULT_TARGET_MAGNIFICATION:
            raise PatchTilerConfigError(
                "target_magnification must be "
                f"{DEFAULT_TARGET_MAGNIFICATION}, got {target_magnification_int}."
            )

        overlap_value: int = patch_size_int - stride_int
        if overlap_value != DEFAULT_OVERLAP:
            raise PatchTilerConfigError(
                f"overlap must be {DEFAULT_OVERLAP}, got {overlap_value}."
            )

        self._patch_size: int = patch_size_int
        self._stride: int = stride_int
        self._target_magnification: int = target_magnification_int
        self._min_tissue_fraction: float = DEFAULT_MIN_TISSUE_FRACTION

    def tile(self, slide: object, tissue_mask: object) -> list[tuple[int, int]]:
        """Generate level-0 patch coordinates intersecting tissue.

        Args:
            slide: Open slide handle from ``src/data/wsi_reader.py``.
            tissue_mask: Either raw 2D mask array or packet mapping containing
                at least key ``mask`` and optionally ``level``/``downsample``.

        Returns:
            Deterministically ordered ``[(x, y), ...]`` in row-major order (y then x).

        Raises:
            PatchTilerMetadataError: If slide metadata is invalid.
            PatchTilerMaskError: If mask payload is invalid.
        """
        slide_meta: _SlideMeta = self._parse_slide_meta(slide)
        mask_meta: _MaskMeta = self._parse_mask_meta(slide_meta=slide_meta, tissue_mask=tissue_mask)

        _, extraction_downsample = self._resolve_target_level(slide_meta)

        patch_extent_level0: int = max(
            1,
            int(round(float(self._patch_size) * float(extraction_downsample))),
        )
        stride_level0: int = max(
            1,
            int(round(float(self._stride) * float(extraction_downsample))),
        )

        max_x_exclusive: int = slide_meta.width_level0 - patch_extent_level0 + 1
        max_y_exclusive: int = slide_meta.height_level0 - patch_extent_level0 + 1

        if max_x_exclusive <= 0 or max_y_exclusive <= 0:
            return []

        coords: List[Tuple[int, int]] = []

        y_value: int
        for y_value in range(0, max_y_exclusive, stride_level0):
            x_value: int
            for x_value in range(0, max_x_exclusive, stride_level0):
                if self._tile_overlaps_tissue(
                    mask_meta=mask_meta,
                    x_level0=x_value,
                    y_level0=y_value,
                    patch_extent_level0=patch_extent_level0,
                ):
                    coords.append((x_value, y_value))

        # Explicit deterministic ordering even if future logic changes iteration.
        coords.sort(key=lambda item: (item[1], item[0]))
        return coords

    def extract(self, slide: object, coords: list[tuple[int, int]]) -> list[bytes]:
        """Extract RGB patch bytes for each coordinate.

        Args:
            slide: Open slide handle from ``src/data/wsi_reader.py``.
            coords: Level-0 top-left coordinates from :meth:`tile`.

        Returns:
            List of RGB bytes with one-to-one index alignment to ``coords``.

        Raises:
            PatchTilerMetadataError: If slide metadata is invalid.
            PatchTilerReadError: If any patch cannot be extracted.
        """
        if not isinstance(coords, list):
            raise PatchTilerReadError(f"coords must be list[tuple[int,int]], got {type(coords).__name__}.")

        slide_meta: _SlideMeta = self._parse_slide_meta(slide)
        extraction_level, extraction_downsample = self._resolve_target_level(slide_meta)
        patch_extent_level0: int = max(
            1,
            int(round(float(self._patch_size) * float(extraction_downsample))),
        )

        patch_bytes_list: List[bytes] = []
        for index, coord in enumerate(coords):
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise PatchTilerReadError(
                    f"coords[{index}] must be tuple[int,int], got {coord!r}."
                )

            x_level0: int = _coerce_non_negative_int(coord[0], f"coords[{index}][0]")
            y_level0: int = _coerce_non_negative_int(coord[1], f"coords[{index}][1]")

            if (x_level0 + patch_extent_level0) > slide_meta.width_level0:
                raise PatchTilerReadError(
                    "Patch exceeds slide width at extraction geometry: "
                    f"index={index}, x={x_level0}, extent={patch_extent_level0}, "
                    f"width={slide_meta.width_level0}."
                )
            if (y_level0 + patch_extent_level0) > slide_meta.height_level0:
                raise PatchTilerReadError(
                    "Patch exceeds slide height at extraction geometry: "
                    f"index={index}, y={y_level0}, extent={patch_extent_level0}, "
                    f"height={slide_meta.height_level0}."
                )

            patch_bytes: bytes = self._read_patch_bytes(
                slide=slide,
                slide_meta=slide_meta,
                x_level0=x_level0,
                y_level0=y_level0,
                level=extraction_level,
            )
            patch_bytes_list.append(patch_bytes)

        return patch_bytes_list

    def _parse_slide_meta(self, slide: object) -> _SlideMeta:
        """Parse and validate required slide metadata from handle."""
        backend_raw: Any = getattr(slide, "backend", None)
        if backend_raw is None:
            raise PatchTilerMetadataError("Slide handle missing attribute 'backend'.")
        backend: str = str(backend_raw).strip().lower()
        if backend not in {"openslide", "tifffile"}:
            raise PatchTilerMetadataError(f"Unsupported slide backend: {backend!r}.")

        level_dimensions_raw: Any = getattr(slide, "level_dimensions", None)
        level_downsamples_raw: Any = getattr(slide, "level_downsamples", None)

        if level_dimensions_raw is None or level_downsamples_raw is None:
            raise PatchTilerMetadataError(
                "Slide handle must expose 'level_dimensions' and 'level_downsamples'."
            )

        level_dimensions: Tuple[Tuple[int, int], ...] = tuple(
            (int(pair[0]), int(pair[1])) for pair in level_dimensions_raw
        )
        level_downsamples: Tuple[float, ...] = tuple(float(item) for item in level_downsamples_raw)

        level_count: int = len(level_dimensions)
        if level_count <= 0:
            raise PatchTilerMetadataError("Slide has no pyramid levels.")
        if len(level_downsamples) != level_count:
            raise PatchTilerMetadataError(
                "Slide metadata mismatch: len(level_downsamples) != len(level_dimensions)."
            )

        width_level0: int = int(level_dimensions[0][0])
        height_level0: int = int(level_dimensions[0][1])
        if width_level0 <= 0 or height_level0 <= 0:
            raise PatchTilerMetadataError(
                f"Invalid level-0 dimensions: {(width_level0, height_level0)}"
            )

        properties: Mapping[str, Any] = getattr(slide, "properties", {})
        objective_power: Optional[float] = _extract_objective_power(properties)
        mpp: Optional[float] = _extract_mpp(properties)

        return _SlideMeta(
            backend=backend,
            width_level0=width_level0,
            height_level0=height_level0,
            level_count=level_count,
            level_downsamples=level_downsamples,
            level_dimensions=level_dimensions,
            objective_power=objective_power,
            mpp=mpp,
        )

    def _parse_mask_meta(self, slide_meta: _SlideMeta, tissue_mask: object) -> _MaskMeta:
        """Normalize tissue mask payload into binary mask + geometric metadata."""
        if isinstance(tissue_mask, Mapping):
            if "mask" not in tissue_mask:
                raise PatchTilerMaskError("Mask packet must contain key 'mask'.")
            mask_array: np.ndarray = self._coerce_binary_mask(tissue_mask["mask"])

            level_value_raw: Any = tissue_mask.get("level", None)
            downsample_value_raw: Any = tissue_mask.get("downsample", None)

            if level_value_raw is None and downsample_value_raw is None:
                level_value: int = 0
                downsample_value: float = 1.0
            elif level_value_raw is not None and downsample_value_raw is None:
                level_value = _coerce_non_negative_int(level_value_raw, "tissue_mask.level")
                if level_value >= slide_meta.level_count:
                    raise PatchTilerMaskError(
                        f"Mask level {level_value} is out of range for slide levels {slide_meta.level_count}."
                    )
                downsample_value = float(slide_meta.level_downsamples[level_value])
            elif level_value_raw is None and downsample_value_raw is not None:
                downsample_value = _coerce_positive_float(
                    downsample_value_raw,
                    "tissue_mask.downsample",
                )
                level_value = self._nearest_level_for_downsample(
                    slide_meta=slide_meta,
                    desired_downsample=downsample_value,
                )
            else:
                level_value = _coerce_non_negative_int(level_value_raw, "tissue_mask.level")
                if level_value >= slide_meta.level_count:
                    raise PatchTilerMaskError(
                        f"Mask level {level_value} is out of range for slide levels {slide_meta.level_count}."
                    )
                downsample_value = _coerce_positive_float(
                    downsample_value_raw,
                    "tissue_mask.downsample",
                )
        else:
            mask_array = self._coerce_binary_mask(tissue_mask)
            level_value = 0
            downsample_value = 1.0

        expected_height: int = int(slide_meta.level_dimensions[level_value][1])
        expected_width: int = int(slide_meta.level_dimensions[level_value][0])

        if mask_array.shape != (expected_height, expected_width):
            # Validate consistency when downsample was provided but level shape mismatches;
            # if level metadata is wrong, infer downsample from shape ratio.
            inferred_downsample_x: float = float(slide_meta.width_level0) / float(mask_array.shape[1])
            inferred_downsample_y: float = float(slide_meta.height_level0) / float(mask_array.shape[0])
            inferred_downsample: float = max(inferred_downsample_x, inferred_downsample_y)

            if not math.isfinite(inferred_downsample) or inferred_downsample <= 0.0:
                raise PatchTilerMaskError(
                    "Unable to infer valid mask downsample from mask shape "
                    f"{mask_array.shape} and slide level-0 dimensions "
                    f"({slide_meta.width_level0}, {slide_meta.height_level0})."
                )

            level_value = self._nearest_level_for_downsample(
                slide_meta=slide_meta,
                desired_downsample=inferred_downsample,
            )
            downsample_value = inferred_downsample

        return _MaskMeta(
            mask=mask_array,
            level=level_value,
            downsample=downsample_value,
        )

    def _coerce_binary_mask(self, mask: object) -> np.ndarray:
        """Coerce mask-like input to 2D uint8 binary array."""
        mask_array: np.ndarray = np.asarray(mask)
        if mask_array.ndim != 2:
            raise PatchTilerMaskError(
                f"Mask must be rank-2 array, got shape {mask_array.shape}."
            )
        if mask_array.size == 0:
            raise PatchTilerMaskError("Mask cannot be empty.")

        if mask_array.dtype == np.bool_:
            return mask_array.astype(np.uint8)

        if np.issubdtype(mask_array.dtype, np.integer):
            return (mask_array > 0).astype(np.uint8)

        if np.issubdtype(mask_array.dtype, np.floating):
            finite_values: np.ndarray = np.nan_to_num(
                mask_array.astype(np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            return (finite_values > 0.5).astype(np.uint8)

        raise PatchTilerMaskError(f"Unsupported mask dtype: {mask_array.dtype}.")

    def _resolve_target_level(self, slide_meta: _SlideMeta) -> Tuple[int, float]:
        """Resolve extraction level closest to target magnification."""
        scan_magnification: Optional[float] = None
        if slide_meta.objective_power is not None and slide_meta.objective_power > 0.0:
            scan_magnification = slide_meta.objective_power
        elif slide_meta.mpp is not None and slide_meta.mpp > 0.0:
            scan_magnification = 10.0 / float(slide_meta.mpp)

        if scan_magnification is None or not math.isfinite(scan_magnification) or scan_magnification <= 0.0:
            desired_downsample: float = 1.0
        else:
            desired_downsample = float(scan_magnification) / float(self._target_magnification)
            if not math.isfinite(desired_downsample) or desired_downsample <= 0.0:
                desired_downsample = 1.0

        level_index: int = self._nearest_level_for_downsample(
            slide_meta=slide_meta,
            desired_downsample=desired_downsample,
        )
        actual_downsample: float = float(slide_meta.level_downsamples[level_index])
        if actual_downsample <= 0.0:
            raise PatchTilerMetadataError(
                f"Resolved non-positive downsample at level {level_index}: {actual_downsample}"
            )

        return level_index, actual_downsample

    def _nearest_level_for_downsample(self, slide_meta: _SlideMeta, desired_downsample: float) -> int:
        """Find level index with minimal absolute downsample error."""
        best_level: int = 0
        best_error: float = float("inf")

        for level_index, downsample_value in enumerate(slide_meta.level_downsamples):
            error_value: float = abs(float(downsample_value) - float(desired_downsample))
            if error_value < best_error:
                best_error = error_value
                best_level = level_index

        return best_level

    def _tile_overlaps_tissue(
        self,
        mask_meta: _MaskMeta,
        x_level0: int,
        y_level0: int,
        patch_extent_level0: int,
    ) -> bool:
        """Return True if tile overlaps tissue beyond minimum fraction."""
        mask_height: int = int(mask_meta.mask.shape[0])
        mask_width: int = int(mask_meta.mask.shape[1])
        downsample: float = float(mask_meta.downsample)

        x0_mask: int = int(math.floor(float(x_level0) / downsample))
        y0_mask: int = int(math.floor(float(y_level0) / downsample))
        x1_mask: int = int(math.ceil(float(x_level0 + patch_extent_level0) / downsample))
        y1_mask: int = int(math.ceil(float(y_level0 + patch_extent_level0) / downsample))

        x0_mask = max(0, min(mask_width, x0_mask))
        y0_mask = max(0, min(mask_height, y0_mask))
        x1_mask = max(0, min(mask_width, x1_mask))
        y1_mask = max(0, min(mask_height, y1_mask))

        if x1_mask <= x0_mask or y1_mask <= y0_mask:
            return False

        tile_mask: np.ndarray = mask_meta.mask[y0_mask:y1_mask, x0_mask:x1_mask]
        if tile_mask.size == 0:
            return False

        tissue_fraction: float = float(np.count_nonzero(tile_mask)) / float(tile_mask.size)
        return tissue_fraction > float(self._min_tissue_fraction)

    def _read_patch_bytes(
        self,
        slide: object,
        slide_meta: _SlideMeta,
        x_level0: int,
        y_level0: int,
        level: int,
    ) -> bytes:
        """Read one patch and return packed RGB bytes."""
        raw_object: Any = getattr(slide, "raw", None)
        if raw_object is None:
            raise PatchTilerReadError("Slide handle missing attribute 'raw'.")

        if slide_meta.backend == "openslide":
            return self._read_patch_bytes_openslide(
                raw_object=raw_object,
                x_level0=x_level0,
                y_level0=y_level0,
                level=level,
            )

        return self._read_patch_bytes_tifffile(
            slide=slide,
            raw_object=raw_object,
            x_level0=x_level0,
            y_level0=y_level0,
            level=level,
            downsample=float(slide_meta.level_downsamples[level]),
        )

    def _read_patch_bytes_openslide(
        self,
        raw_object: object,
        x_level0: int,
        y_level0: int,
        level: int,
    ) -> bytes:
        """Read patch via OpenSlide backend."""
        try:
            region: Any = raw_object.read_region(
                (int(x_level0), int(y_level0)),
                int(level),
                (int(self._patch_size), int(self._patch_size)),
            )
        except Exception as exc:  # noqa: BLE001
            raise PatchTilerReadError(
                "OpenSlide read_region failed for "
                f"(x={x_level0}, y={y_level0}, level={level}): {exc}"
            ) from exc

        if hasattr(region, "convert"):
            region = region.convert("RGB")

        region_array: np.ndarray = np.asarray(region)
        if region_array.ndim != 3 or region_array.shape[2] != 3:
            raise PatchTilerReadError(
                f"OpenSlide returned invalid patch shape {region_array.shape}."
            )

        if region_array.shape[0] != self._patch_size or region_array.shape[1] != self._patch_size:
            raise PatchTilerReadError(
                "OpenSlide returned unexpected patch spatial size "
                f"{region_array.shape[:2]}, expected ({self._patch_size}, {self._patch_size})."
            )

        if region_array.dtype != np.uint8:
            region_array = region_array.astype(np.uint8, copy=False)

        return region_array.tobytes(order="C")

    def _read_patch_bytes_tifffile(
        self,
        slide: object,
        raw_object: object,
        x_level0: int,
        y_level0: int,
        level: int,
        downsample: float,
    ) -> bytes:
        """Read patch via tifffile backend."""
        level_array: np.ndarray = self._get_tiff_level_array(
            slide=slide,
            raw_object=raw_object,
            level=level,
        )

        if level_array.ndim == 2:
            level_array = np.stack([level_array, level_array, level_array], axis=-1)
        elif level_array.ndim == 3 and level_array.shape[2] >= 3:
            level_array = level_array[:, :, :3]
        else:
            raise PatchTilerReadError(
                f"Unsupported tifffile level array shape {level_array.shape} at level {level}."
            )

        if level_array.dtype != np.uint8:
            level_array = _to_uint8(level_array)

        x_level: int = int(math.floor(float(x_level0) / downsample))
        y_level: int = int(math.floor(float(y_level0) / downsample))

        x_end: int = x_level + self._patch_size
        y_end: int = y_level + self._patch_size

        if x_level < 0 or y_level < 0:
            raise PatchTilerReadError(
                f"Negative level coordinates computed: {(x_level, y_level)}."
            )
        if y_end > level_array.shape[0] or x_end > level_array.shape[1]:
            raise PatchTilerReadError(
                "Requested tifffile patch exceeds level bounds: "
                f"level={level}, coord=({x_level},{y_level}), "
                f"patch_size={self._patch_size}, level_shape={level_array.shape[:2]}."
            )

        patch_array: np.ndarray = level_array[y_level:y_end, x_level:x_end, :]
        if patch_array.shape != (self._patch_size, self._patch_size, 3):
            raise PatchTilerReadError(
                f"Unexpected tifffile patch shape {patch_array.shape}."
            )

        return patch_array.tobytes(order="C")

    def _get_tiff_level_array(self, slide: object, raw_object: object, level: int) -> np.ndarray:
        """Load/cached tifffile level image array."""
        level_cache: Any = getattr(slide, "level_cache", None)
        if isinstance(level_cache, dict) and level in level_cache:
            return np.asarray(level_cache[level])

        try:
            series_object: Any = raw_object.series[0]
            levels_raw: Any = getattr(series_object, "levels", [series_object])
            level_object: Any = levels_raw[int(level)]
            level_array: np.ndarray = np.asarray(level_object.asarray())
        except Exception as exc:  # noqa: BLE001
            raise PatchTilerReadError(
                f"Failed loading tifffile level {level}: {exc}"
            ) from exc

        if isinstance(level_cache, dict):
            level_cache[level] = level_array

        return level_array

def _coerce_positive_int(value: Any, field_name: str) -> int:
    """Coerce positive integer with strict validation."""
    if isinstance(value, bool):
        raise PatchTilerConfigError(f"{field_name} must be integer, got bool.")
    try:
        value_int: int = int(value)
    except Exception as exc:  # noqa: BLE001
        raise PatchTilerConfigError(f"{field_name} must be integer, got {value!r}.") from exc
    if value_int <= 0:
        raise PatchTilerConfigError(f"{field_name} must be > 0, got {value_int}.")
    return value_int


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    """Coerce non-negative integer with strict validation."""
    if isinstance(value, bool):
        raise PatchTilerReadError(f"{field_name} must be integer, got bool.")
    try:
        value_int: int = int(value)
    except Exception as exc:  # noqa: BLE001
        raise PatchTilerReadError(f"{field_name} must be integer, got {value!r}.") from exc
    if value_int < 0:
        raise PatchTilerReadError(f"{field_name} must be >= 0, got {value_int}.")
    return value_int


def _coerce_positive_float(value: Any, field_name: str) -> float:
    """Coerce positive finite float with strict validation."""
    if isinstance(value, bool):
        raise PatchTilerMaskError(f"{field_name} must be float, got bool.")
    try:
        value_float: float = float(value)
    except Exception as exc:  # noqa: BLE001
        raise PatchTilerMaskError(f"{field_name} must be float, got {value!r}.") from exc

    if not math.isfinite(value_float) or value_float <= 0.0:
        raise PatchTilerMaskError(
            f"{field_name} must be finite and > 0, got {value_float}."
        )
    return value_float


def _extract_objective_power(properties: Mapping[str, Any]) -> Optional[float]:
    """Extract objective magnification from known metadata keys."""
    if not isinstance(properties, Mapping):
        return None

    candidate_keys: Tuple[str, ...] = (
        "openslide.objective-power",
        "openslide.objective_power",
        "aperio.AppMag",
        "objective_power",
        "objective",
    )
    for key in candidate_keys:
        if key not in properties:
            continue
        parsed: Optional[float] = _safe_float(properties[key])
        if parsed is not None and parsed > 0.0:
            return parsed
    return None


def _extract_mpp(properties: Mapping[str, Any]) -> Optional[float]:
    """Extract level-0 MPP from known metadata keys."""
    if not isinstance(properties, Mapping):
        return None

    candidate_keys: Tuple[str, ...] = (
        "openslide.mpp-x",
        "openslide.mpp_x",
        "aperio.MPP",
        "mpp",
        "mpp_x",
    )
    for key in candidate_keys:
        if key not in properties:
            continue
        parsed: Optional[float] = _safe_float(properties[key])
        if parsed is not None and parsed > 0.0:
            return parsed
    return None


def _safe_float(value: Any) -> Optional[float]:
    """Best-effort finite float parser."""
    try:
        parsed: float = float(str(value).strip())
    except Exception:  # noqa: BLE001
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert numeric array to uint8 deterministically."""
    if array.dtype == np.uint8:
        return array

    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8)

    float_array: np.ndarray = np.asarray(array, dtype=np.float32)
    float_array = np.nan_to_num(float_array, nan=0.0, posinf=0.0, neginf=0.0)

    min_value: float = float(np.min(float_array))
    max_value: float = float(np.max(float_array))

    if max_value <= min_value:
        return np.zeros_like(float_array, dtype=np.uint8)

    normalized: np.ndarray = (float_array - min_value) / (max_value - min_value)
    scaled: np.ndarray = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)


__all__ = [
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_PATCH_STRIDE",
    "DEFAULT_TARGET_MAGNIFICATION",
    "DEFAULT_OVERLAP",
    "DEFAULT_MIN_TISSUE_FRACTION",
    "PatchTilerError",
    "PatchTilerConfigError",
    "PatchTilerMetadataError",
    "PatchTilerMaskError",
    "PatchTilerReadError",
    "PatchTiler",
]
