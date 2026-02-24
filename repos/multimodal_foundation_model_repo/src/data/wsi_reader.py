"""Whole-slide image (WSI) reader utilities.

This module provides the design-locked ``WSIReader`` interface:
- ``__init__(backend: str) -> None``
- ``open(slide_path: str) -> object``
- ``read_region(slide: object, x: int, y: int, w: int, h: int, level: int) -> bytes``
- ``get_mpp(slide: object) -> float``
- ``close(slide: object) -> None``

Implementation is aligned with paper/config preprocessing assumptions:
- target magnification: 20x
- patch geometry defaults in pipeline: 512x512, stride 512, no overlap

Notes:
- ``read_region`` returns raw RGB bytes in row-major order with shape ``(h, w, 3)``.
- Coordinates ``(x, y)`` are interpreted in level-0 reference frame.
- ``w`` and ``h`` are interpreted in requested ``level`` pixel units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import openslide
except Exception:  # pragma: no cover - environment dependent import
    openslide = None  # type: ignore[assignment]

try:
    import tifffile
except Exception:  # pragma: no cover - environment dependent import
    tifffile = None  # type: ignore[assignment]


DEFAULT_BACKEND: str = "openslide"
DEFAULT_TARGET_MAGNIFICATION_X: float = 20.0
DEFAULT_TARGET_MPP_UM_PER_PX: float = 0.5
DEFAULT_PATCH_SIZE: int = 512
DEFAULT_PATCH_STRIDE: int = 512
DEFAULT_OVERLAP: int = 0
DEFAULT_OOB_PAD_VALUE: int = 255


class WSIReaderError(Exception):
    """Base exception for WSI reader failures."""


class WSIBackendError(WSIReaderError):
    """Raised when backend configuration or backend calls fail."""


class WSIPathError(WSIReaderError):
    """Raised when slide path is missing or invalid."""


class WSIMetadataError(WSIReaderError):
    """Raised when required WSI metadata is missing or inconsistent."""


class WSIReadError(WSIReaderError):
    """Raised when region extraction fails."""


@dataclass(slots=True)
class _SlideHandle:
    """Opaque internal handle for opened slide resources."""

    backend: str
    path: str
    raw: Any
    level_count: int
    level_dimensions: List[Tuple[int, int]]
    level_downsamples: List[float]
    properties: Dict[str, str] = field(default_factory=dict)
    closed: bool = False
    # Only used for tifffile backend. Cached level arrays avoid repeated decoding.
    level_cache: Dict[int, np.ndarray] = field(default_factory=dict)


class WSIReader:
    """Magnification-aware WSI adapter.

    Args:
        backend: Backend key. Supported values:
            - ``"openslide"`` (default)
            - ``"tifffile"``
    """

    def __init__(self, backend: str = DEFAULT_BACKEND) -> None:
        normalized_backend: str = str(backend).strip().lower()
        if normalized_backend not in {"openslide", "tifffile"}:
            raise WSIBackendError(
                "Unsupported backend. Expected one of {'openslide', 'tifffile'}, "
                f"got {backend!r}."
            )
        self._backend: str = normalized_backend

    def open(self, slide_path: str) -> object:
        """Open a whole-slide image and return an opaque slide handle.

        Args:
            slide_path: Filesystem path to WSI.

        Returns:
            Opaque slide handle object.

        Raises:
            WSIPathError: If path does not exist or is not a file.
            WSIBackendError: If backend open fails.
        """
        normalized_path: Path = Path(str(slide_path)).expanduser().resolve()
        if not normalized_path.exists():
            raise WSIPathError(f"Slide does not exist: {normalized_path}")
        if not normalized_path.is_file():
            raise WSIPathError(f"Slide path is not a file: {normalized_path}")

        if self._backend == "openslide":
            return self._open_with_openslide(normalized_path)
        return self._open_with_tifffile(normalized_path)

    def read_region(
        self,
        slide: object,
        x: int,
        y: int,
        w: int,
        h: int,
        level: int,
    ) -> bytes:
        """Read a rectangular region and return raw RGB bytes.

        Args:
            slide: Opaque handle returned by :meth:`open`.
            x: Top-left x in level-0 coordinates.
            y: Top-left y in level-0 coordinates.
            w: Region width in pixels at ``level``.
            h: Region height in pixels at ``level``.
            level: Pyramid level index.

        Returns:
            RGB bytes of length ``w * h * 3``.

        Raises:
            WSIReadError: For invalid arguments or backend read failures.
        """
        handle: _SlideHandle = self._validate_handle(slide)

        x_int: int = _coerce_non_negative_int(x, "x")
        y_int: int = _coerce_non_negative_int(y, "y")
        w_int: int = _coerce_positive_int(w, "w")
        h_int: int = _coerce_positive_int(h, "h")
        level_int: int = _coerce_non_negative_int(level, "level")

        if level_int >= handle.level_count:
            raise WSIReadError(
                f"Invalid level={level_int}; valid range is [0, {handle.level_count - 1}]."
            )

        try:
            if handle.backend == "openslide":
                rgb_array: np.ndarray = self._read_region_openslide(
                    handle=handle,
                    x=x_int,
                    y=y_int,
                    w=w_int,
                    h=h_int,
                    level=level_int,
                )
            else:
                rgb_array = self._read_region_tifffile(
                    handle=handle,
                    x=x_int,
                    y=y_int,
                    w=w_int,
                    h=h_int,
                    level=level_int,
                )
        except WSIReadError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WSIReadError(
                "Failed to read WSI region "
                f"(path={handle.path}, x={x_int}, y={y_int}, w={w_int}, h={h_int}, level={level_int}): {exc}"
            ) from exc

        if rgb_array.shape != (h_int, w_int, 3):
            raise WSIReadError(
                "Region reader returned invalid shape "
                f"{rgb_array.shape}; expected ({h_int}, {w_int}, 3)."
            )

        if rgb_array.dtype != np.uint8:
            rgb_array = rgb_array.astype(np.uint8, copy=False)

        return rgb_array.tobytes(order="C")

    def get_mpp(self, slide: object) -> float:
        """Return microns-per-pixel (MPP) estimate for level-0.

        Resolution policy:
        1. Use explicit MPP metadata if available.
        2. Fallback to objective power conversion: ``mpp = 10 / objective_power``.

        Args:
            slide: Opaque handle returned by :meth:`open`.

        Returns:
            Positive MPP value in microns per pixel.

        Raises:
            WSIMetadataError: If MPP cannot be resolved.
        """
        handle: _SlideHandle = self._validate_handle(slide)
        properties: Mapping[str, str] = handle.properties

        mpp_value: Optional[float] = self._parse_first_float_property(
            properties=properties,
            keys=self._mpp_candidate_keys(),
        )
        if mpp_value is not None and mpp_value > 0.0:
            return float(mpp_value)

        objective_power: Optional[float] = self._parse_first_float_property(
            properties=properties,
            keys=self._objective_candidate_keys(),
        )
        if objective_power is not None and objective_power > 0.0:
            inferred_mpp: float = 10.0 / float(objective_power)
            if inferred_mpp > 0.0:
                return inferred_mpp

        raise WSIMetadataError(
            "Unable to resolve MPP from slide metadata. "
            f"path={handle.path}, backend={handle.backend}"
        )

    def close(self, slide: object) -> None:
        """Close an opened slide handle.

        Args:
            slide: Opaque handle returned by :meth:`open`.

        Raises:
            WSIBackendError: If close operation fails.
        """
        handle: _SlideHandle = self._validate_handle(slide, allow_closed=True)
        if handle.closed:
            return

        try:
            raw_object: Any = handle.raw
            if hasattr(raw_object, "close") and callable(raw_object.close):
                raw_object.close()
            handle.level_cache.clear()
            handle.closed = True
        except Exception as exc:  # noqa: BLE001
            raise WSIBackendError(
                f"Failed to close slide handle for path={handle.path}: {exc}"
            ) from exc

    def _open_with_openslide(self, slide_path: Path) -> _SlideHandle:
        """Open slide with OpenSlide backend."""
        if openslide is None:
            raise WSIBackendError(
                "OpenSlide backend requested but openslide-python is unavailable."
            )

        try:
            slide_object: Any = openslide.OpenSlide(str(slide_path))
        except Exception as exc:  # noqa: BLE001
            raise WSIBackendError(
                f"OpenSlide failed opening {slide_path}: {exc}"
            ) from exc

        level_count: int = int(getattr(slide_object, "level_count", 0))
        level_dimensions_raw: Sequence[Tuple[int, int]] = getattr(
            slide_object,
            "level_dimensions",
            (),
        )
        level_dimensions: List[Tuple[int, int]] = [
            (int(width), int(height)) for width, height in level_dimensions_raw
        ]
        level_downsamples_raw: Sequence[float] = getattr(
            slide_object,
            "level_downsamples",
            (),
        )
        level_downsamples: List[float] = [float(value) for value in level_downsamples_raw]

        if level_count <= 0 or len(level_dimensions) != level_count:
            # Build best-effort dimensions from level 0 if partially missing.
            if len(level_dimensions) == 0 and hasattr(slide_object, "dimensions"):
                width0, height0 = getattr(slide_object, "dimensions")
                level_dimensions = [(int(width0), int(height0))]
                level_count = 1
                level_downsamples = [1.0]
            else:
                self._safe_close_raw(slide_object)
                raise WSIBackendError(
                    f"Invalid OpenSlide pyramid metadata for {slide_path}."
                )

        if len(level_downsamples) != level_count:
            level_downsamples = _infer_downsamples_from_dimensions(level_dimensions)

        properties_raw: Dict[str, str] = dict(getattr(slide_object, "properties", {}))

        return _SlideHandle(
            backend="openslide",
            path=str(slide_path),
            raw=slide_object,
            level_count=level_count,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
            properties={str(key): str(value) for key, value in properties_raw.items()},
        )

    def _open_with_tifffile(self, slide_path: Path) -> _SlideHandle:
        """Open slide with tifffile backend."""
        if tifffile is None:
            raise WSIBackendError(
                "tifffile backend requested but tifffile is unavailable."
            )

        try:
            tiff_object: Any = tifffile.TiffFile(str(slide_path))
        except Exception as exc:  # noqa: BLE001
            raise WSIBackendError(
                f"tifffile failed opening {slide_path}: {exc}"
            ) from exc

        try:
            series_object: Any = tiff_object.series[0]
            levels_raw: Sequence[Any] = getattr(series_object, "levels", [series_object])
            if len(levels_raw) == 0:
                levels_raw = [series_object]

            level_dimensions: List[Tuple[int, int]] = []
            for level_object in levels_raw:
                shape: Tuple[int, ...] = tuple(int(item) for item in level_object.shape)
                if len(shape) < 2:
                    raise WSIBackendError(
                        f"Unsupported tifffile level shape {shape} for {slide_path}."
                    )
                height: int = int(shape[0])
                width: int = int(shape[1])
                level_dimensions.append((width, height))

            level_count: int = len(level_dimensions)
            level_downsamples: List[float] = _infer_downsamples_from_dimensions(level_dimensions)

            properties: Dict[str, str] = {}
            if hasattr(tiff_object, "pages") and len(tiff_object.pages) > 0:
                first_page: Any = tiff_object.pages[0]
                tags: Any = getattr(first_page, "tags", {})
                for key_name in ("XResolution", "YResolution", "ResolutionUnit"):
                    if key_name in tags:
                        properties[f"tiff.{key_name}"] = str(tags[key_name].value)

            return _SlideHandle(
                backend="tifffile",
                path=str(slide_path),
                raw=tiff_object,
                level_count=level_count,
                level_dimensions=level_dimensions,
                level_downsamples=level_downsamples,
                properties=properties,
            )
        except Exception:
            self._safe_close_raw(tiff_object)
            raise

    def _read_region_openslide(
        self,
        handle: _SlideHandle,
        x: int,
        y: int,
        w: int,
        h: int,
        level: int,
    ) -> np.ndarray:
        """Read region through OpenSlide and return RGB uint8 array."""
        try:
            region_image: Any = handle.raw.read_region((x, y), level, (w, h))
        except Exception as exc:  # noqa: BLE001
            raise WSIReadError(
                f"OpenSlide read_region failed for {handle.path}: {exc}"
            ) from exc

        if hasattr(region_image, "convert"):
            region_image = region_image.convert("RGB")

        region_array: np.ndarray = np.asarray(region_image, dtype=np.uint8)
        if region_array.ndim != 3 or region_array.shape[2] != 3:
            raise WSIReadError(
                f"OpenSlide returned invalid region shape {region_array.shape}."
            )
        return region_array

    def _read_region_tifffile(
        self,
        handle: _SlideHandle,
        x: int,
        y: int,
        w: int,
        h: int,
        level: int,
    ) -> np.ndarray:
        """Read region through tifffile backend and return RGB uint8 array."""
        level_array: np.ndarray = self._get_tiff_level_array(handle=handle, level=level)
        if level_array.ndim == 2:
            level_array = np.stack([level_array, level_array, level_array], axis=-1)
        elif level_array.ndim == 3 and level_array.shape[2] == 4:
            level_array = level_array[:, :, :3]
        elif level_array.ndim != 3 or level_array.shape[2] not in {1, 3}:
            raise WSIReadError(
                f"Unsupported tifffile level array shape {level_array.shape} for {handle.path}."
            )

        if level_array.shape[2] == 1:
            level_array = np.repeat(level_array, repeats=3, axis=2)

        downsample: float = float(handle.level_downsamples[level])
        if downsample <= 0.0:
            raise WSIReadError(
                f"Invalid downsample value {downsample} at level {level} for {handle.path}."
            )

        x_level: int = int(math.floor(float(x) / downsample))
        y_level: int = int(math.floor(float(y) / downsample))

        level_width: int = int(level_array.shape[1])
        level_height: int = int(level_array.shape[0])

        out_array: np.ndarray = np.full(
            shape=(h, w, 3),
            fill_value=DEFAULT_OOB_PAD_VALUE,
            dtype=np.uint8,
        )

        src_x0: int = max(0, x_level)
        src_y0: int = max(0, y_level)
        src_x1: int = min(level_width, x_level + w)
        src_y1: int = min(level_height, y_level + h)

        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return out_array

        dst_x0: int = max(0, -x_level)
        dst_y0: int = max(0, -y_level)
        dst_x1: int = dst_x0 + (src_x1 - src_x0)
        dst_y1: int = dst_y0 + (src_y1 - src_y0)

        out_array[dst_y0:dst_y1, dst_x0:dst_x1, :] = level_array[src_y0:src_y1, src_x0:src_x1, :]
        return out_array

    def _get_tiff_level_array(self, handle: _SlideHandle, level: int) -> np.ndarray:
        """Load and cache tifffile level pixel array."""
        if level in handle.level_cache:
            return handle.level_cache[level]

        raw_tiff: Any = handle.raw
        try:
            series_object: Any = raw_tiff.series[0]
            levels_raw: Sequence[Any] = getattr(series_object, "levels", [series_object])
            level_object: Any = levels_raw[level]
            level_array: np.ndarray = np.asarray(level_object.asarray())
        except Exception as exc:  # noqa: BLE001
            raise WSIReadError(
                f"Failed loading tifffile level {level} for {handle.path}: {exc}"
            ) from exc

        if level_array.dtype != np.uint8:
            level_array = _to_uint8(level_array)

        handle.level_cache[level] = level_array
        return level_array

    def _validate_handle(self, slide: object, allow_closed: bool = False) -> _SlideHandle:
        """Validate user-facing slide handle object."""
        if not isinstance(slide, _SlideHandle):
            raise WSIReadError(
                f"Invalid slide handle type {type(slide).__name__}; expected internal handle object."
            )
        if slide.backend != self._backend:
            raise WSIReadError(
                f"Handle backend mismatch: handle={slide.backend}, reader={self._backend}."
            )
        if slide.closed and not allow_closed:
            raise WSIReadError(f"Slide handle already closed: {slide.path}")
        return slide

    def _safe_close_raw(self, raw_object: Any) -> None:
        """Close raw backend object without raising."""
        try:
            if hasattr(raw_object, "close") and callable(raw_object.close):
                raw_object.close()
        except Exception:
            return

    def _mpp_candidate_keys(self) -> List[str]:
        """Return metadata keys searched for explicit MPP."""
        candidate_keys: List[str] = [
            "openslide.mpp-x",
            "openslide.mpp_x",
            "aperio.MPP",
            "hamamatsu.XResolution",
            "mpp",
            "mpp_x",
        ]
        if openslide is not None and hasattr(openslide, "PROPERTY_NAME_MPP_X"):
            property_name: Any = getattr(openslide, "PROPERTY_NAME_MPP_X")
            if isinstance(property_name, str) and property_name:
                candidate_keys.insert(0, property_name)
        return candidate_keys

    def _objective_candidate_keys(self) -> List[str]:
        """Return metadata keys searched for objective magnification."""
        candidate_keys: List[str] = [
            "openslide.objective-power",
            "openslide.objective_power",
            "aperio.AppMag",
            "objective_power",
            "objective",
        ]
        if openslide is not None and hasattr(openslide, "PROPERTY_NAME_OBJECTIVE_POWER"):
            property_name: Any = getattr(openslide, "PROPERTY_NAME_OBJECTIVE_POWER")
            if isinstance(property_name, str) and property_name:
                candidate_keys.insert(0, property_name)
        return candidate_keys

    def _parse_first_float_property(
        self,
        properties: Mapping[str, str],
        keys: Sequence[str],
    ) -> Optional[float]:
        """Parse first valid float from candidate metadata keys."""
        for key in keys:
            if key not in properties:
                continue
            raw_value: str = str(properties[key]).strip()
            if not raw_value:
                continue
            try:
                value: float = float(raw_value)
                if math.isfinite(value):
                    return value
            except ValueError:
                # Try comma-separated values (e.g., "0.25,0.25").
                parts: List[str] = [item.strip() for item in raw_value.split(",") if item.strip()]
                for part in parts:
                    try:
                        value = float(part)
                    except ValueError:
                        continue
                    if math.isfinite(value):
                        return value
        return None


def _coerce_positive_int(value: Any, field_name: str) -> int:
    """Convert value to positive int with explicit validation."""
    if isinstance(value, bool):
        raise WSIReadError(f"{field_name} must be an integer, got bool.")
    try:
        value_int: int = int(value)
    except Exception as exc:  # noqa: BLE001
        raise WSIReadError(f"{field_name} must be an integer, got {value!r}.") from exc
    if value_int <= 0:
        raise WSIReadError(f"{field_name} must be > 0, got {value_int}.")
    return value_int


def _coerce_non_negative_int(value: Any, field_name: str) -> int:
    """Convert value to non-negative int with explicit validation."""
    if isinstance(value, bool):
        raise WSIReadError(f"{field_name} must be an integer, got bool.")
    try:
        value_int: int = int(value)
    except Exception as exc:  # noqa: BLE001
        raise WSIReadError(f"{field_name} must be an integer, got {value!r}.") from exc
    if value_int < 0:
        raise WSIReadError(f"{field_name} must be >= 0, got {value_int}.")
    return value_int


def _infer_downsamples_from_dimensions(
    level_dimensions: Sequence[Tuple[int, int]],
) -> List[float]:
    """Infer per-level downsample factors from pyramid dimensions."""
    if not level_dimensions:
        return [1.0]

    width0: float = float(level_dimensions[0][0])
    height0: float = float(level_dimensions[0][1])
    if width0 <= 0.0 or height0 <= 0.0:
        return [1.0 for _ in level_dimensions]

    downsamples: List[float] = []
    for width_level, height_level in level_dimensions:
        width_value: float = float(width_level)
        height_value: float = float(height_level)
        if width_value <= 0.0 or height_value <= 0.0:
            downsamples.append(1.0)
            continue
        width_ratio: float = width0 / width_value
        height_ratio: float = height0 / height_value
        downsample: float = max(width_ratio, height_ratio)
        if not math.isfinite(downsample) or downsample <= 0.0:
            downsample = 1.0
        downsamples.append(downsample)

    if downsamples:
        downsamples[0] = 1.0
    return downsamples


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert arbitrary numeric image array to uint8 deterministically."""
    if array.dtype == np.uint8:
        return array

    casted: np.ndarray = np.asarray(array)
    if casted.size == 0:
        return casted.astype(np.uint8)

    if np.issubdtype(casted.dtype, np.integer):
        clipped: np.ndarray = np.clip(casted, 0, 255)
        return clipped.astype(np.uint8)

    finite_mask: np.ndarray = np.isfinite(casted)
    if not finite_mask.any():
        return np.zeros_like(casted, dtype=np.uint8)

    finite_values: np.ndarray = casted[finite_mask].astype(np.float64, copy=False)
    min_value: float = float(np.min(finite_values))
    max_value: float = float(np.max(finite_values))

    if max_value <= min_value:
        filled: np.ndarray = np.zeros_like(casted, dtype=np.uint8)
        return filled

    normalized: np.ndarray = (casted.astype(np.float64, copy=False) - min_value) / (max_value - min_value)
    normalized = np.clip(normalized, 0.0, 1.0)
    scaled: np.ndarray = np.rint(normalized * 255.0)
    return scaled.astype(np.uint8)


__all__ = [
    "DEFAULT_BACKEND",
    "DEFAULT_TARGET_MAGNIFICATION_X",
    "DEFAULT_TARGET_MPP_UM_PER_PX",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_PATCH_STRIDE",
    "DEFAULT_OVERLAP",
    "DEFAULT_OOB_PAD_VALUE",
    "WSIReaderError",
    "WSIBackendError",
    "WSIPathError",
    "WSIMetadataError",
    "WSIReadError",
    "WSIReader",
]
