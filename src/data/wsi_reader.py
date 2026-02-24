"""Whole-slide image reader abstraction for TITAN reproduction.

This module provides a strict OpenSlide-backed implementation of the design
contract:
- ``WSIReader.__init__(backend: str = "openslide")``
- ``WSIReader.open(path: str) -> Any``
- ``WSIReader.read_region(path: str, x: int, y: int, size: int, level: int) -> np.ndarray``
- ``WSIReader.get_mpp(path: str) -> float``
- ``WSIReader.get_dimensions(path: str) -> tuple[int, int]``

Coordinate convention is level-0 (full-resolution) top-left pixel coordinates.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
except Exception as exc:  # pragma: no cover - import-time environment dependent
    openslide = None  # type: ignore[assignment]
    _OPENSLIDE_IMPORT_ERROR: Optional[Exception] = exc
else:
    _OPENSLIDE_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Paper/config-locked constants.
# -----------------------------------------------------------------------------
_DEFAULT_BACKEND: Final[str] = "openslide"
_PATCH_SIZE_PX: Final[int] = 512
_MAGNIFICATION: Final[str] = "20x"
_FEATURE_DIM: Final[int] = 768

_DEFAULT_MPP_FALLBACK_FROM_OBJECTIVE_FACTOR: Final[float] = 10.0
_ALLOWED_IMAGE_MODE: Final[str] = "RGB"

# Common OpenSlide metadata keys.
_PROP_MPP_X: Final[str] = "openslide.mpp-x"
_PROP_MPP_Y: Final[str] = "openslide.mpp-y"
_PROP_OBJECTIVE_POWER: Final[str] = "openslide.objective-power"

# Vendor-style fallback keys occasionally seen in SVS/NDPI metadata.
_VENDOR_MPP_KEYS_X: Final[Tuple[str, ...]] = (
    _PROP_MPP_X,
    "aperio.MPP",
    "hamamatsu.XResolution",
    "tiff.XResolution",
)
_VENDOR_MPP_KEYS_Y: Final[Tuple[str, ...]] = (
    _PROP_MPP_Y,
    "aperio.MPP",
    "hamamatsu.YResolution",
    "tiff.YResolution",
)


class WSIReaderError(RuntimeError):
    """Base error for WSI reader failures."""


class WSIBackendError(WSIReaderError):
    """Raised when the configured backend is unsupported or unavailable."""


class WSIMetadataError(WSIReaderError):
    """Raised when required WSI metadata cannot be resolved."""


class WSIReader:
    """OpenSlide wrapper for deterministic WSI access.

    Notes:
        - Coordinates are interpreted in level-0 reference frame.
        - ``read_region`` returns ``np.uint8`` RGB arrays of exact shape
          ``(size, size, 3)``.
        - Slide handles are cached per process for performance and closed at
          interpreter exit.
    """

    def __init__(self, backend: str = _DEFAULT_BACKEND) -> None:
        """Initialize the reader.

        Args:
            backend: Storage backend name. Only ``"openslide"`` is supported.

        Raises:
            WSIBackendError: If backend is unsupported or OpenSlide is missing.
        """
        normalized_backend: str = str(backend).strip().lower()
        if normalized_backend != _DEFAULT_BACKEND:
            raise WSIBackendError(
                f"Unsupported backend '{backend}'. Only '{_DEFAULT_BACKEND}' is supported."
            )

        if openslide is None:
            detail: str = (
                f" ({_OPENSLIDE_IMPORT_ERROR})" if _OPENSLIDE_IMPORT_ERROR is not None else ""
            )
            raise WSIBackendError(
                "OpenSlide backend is unavailable. Ensure 'openslide-python==1.2.0' "
                "and system OpenSlide libraries are installed."
                f"{detail}"
            )

        self._backend: str = normalized_backend
        self._slides: Dict[str, Any] = {}
        self._lock: Lock = Lock()
        self._registered_atexit: bool = False

        # Keep config-locked constants attached for provenance/introspection.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM

    def open(self, path: str) -> Any:
        """Open a slide and return the backend handle.

        Args:
            path: Path to WSI file.

        Returns:
            OpenSlide slide handle.

        Raises:
            FileNotFoundError: If the slide path does not exist.
            WSIReaderError: If OpenSlide cannot open the slide.
        """
        slide_path: str = self._normalize_existing_path(path)

        with self._lock:
            cached: Optional[Any] = self._slides.get(slide_path)
            if cached is not None:
                return cached

            try:
                slide: Any = openslide.OpenSlide(slide_path)
            except Exception as exc:
                raise WSIReaderError(f"Failed to open slide '{slide_path}': {exc}") from exc

            self._slides[slide_path] = slide
            if not self._registered_atexit:
                atexit.register(self.close_all)
                self._registered_atexit = True

            return slide

    def read_region(self, path: str, x: int, y: int, size: int, level: int) -> np.ndarray:
        """Read a square RGB crop from a WSI.

        Args:
            path: Path to WSI file.
            x: Level-0 x coordinate of top-left corner.
            y: Level-0 y coordinate of top-left corner.
            size: Output crop size in pixels at requested level.
            level: Pyramid level index.

        Returns:
            A ``uint8`` RGB ndarray of shape ``(size, size, 3)``.

        Raises:
            ValueError: If coordinates/size/level are invalid.
            WSIReaderError: If reading fails.
        """
        if x < 0 or y < 0:
            raise ValueError(f"x and y must be >= 0. Got x={x}, y={y}.")
        if size <= 0:
            raise ValueError(f"size must be > 0. Got size={size}.")
        if level < 0:
            raise ValueError(f"level must be >= 0. Got level={level}.")

        slide: Any = self.open(path)
        level_count: int = int(getattr(slide, "level_count", 0))
        if level_count <= 0:
            raise WSIReaderError(f"Slide has invalid level_count={level_count} for path '{path}'.")
        if level >= level_count:
            raise ValueError(
                f"Requested level={level} out of range for slide '{path}'. "
                f"Valid levels: [0, {level_count - 1}]."
            )

        try:
            pil_region: Image.Image = slide.read_region((int(x), int(y)), int(level), (int(size), int(size)))
        except Exception as exc:
            raise WSIReaderError(
                f"Failed to read region from '{path}' at (x={x}, y={y}, size={size}, level={level}): {exc}"
            ) from exc

        # OpenSlide returns RGBA; enforce deterministic RGB output.
        if pil_region.mode != _ALLOWED_IMAGE_MODE:
            pil_region = pil_region.convert(_ALLOWED_IMAGE_MODE)

        region_np: np.ndarray = np.asarray(pil_region, dtype=np.uint8)

        if region_np.ndim != 3 or region_np.shape[2] != 3:
            raise WSIReaderError(
                f"Unexpected read_region output shape {region_np.shape} for '{path}'. "
                "Expected (H, W, 3)."
            )

        expected_shape: Tuple[int, int, int] = (int(size), int(size), 3)
        if tuple(region_np.shape) != expected_shape:
            # Keep strict behavior to avoid silent downstream drift.
            raise WSIReaderError(
                f"Unexpected crop shape {region_np.shape} for '{path}'. Expected {expected_shape}."
            )

        return region_np

    def get_mpp(self, path: str) -> float:
        """Resolve microns-per-pixel (MPP) from slide metadata.

        Resolution strategy:
            1. Use OpenSlide MPP X/Y properties when available.
            2. Try vendor fallback keys.
            3. Fallback to objective power using ``mpp = 10.0 / objective_power``.

        Args:
            path: Path to WSI file.

        Returns:
            Scalar MPP value in microns-per-pixel.

        Raises:
            WSIMetadataError: If MPP cannot be inferred.
        """
        slide: Any = self.open(path)
        props: Dict[str, str] = dict(getattr(slide, "properties", {}))

        mpp_x: Optional[float] = self._find_float_property(props, _VENDOR_MPP_KEYS_X)
        mpp_y: Optional[float] = self._find_float_property(props, _VENDOR_MPP_KEYS_Y)

        if mpp_x is not None and mpp_y is not None:
            return float((mpp_x + mpp_y) / 2.0)
        if mpp_x is not None:
            return float(mpp_x)
        if mpp_y is not None:
            return float(mpp_y)

        objective_power: Optional[float] = self._find_float_property(props, (_PROP_OBJECTIVE_POWER,))
        if objective_power is not None and objective_power > 0.0:
            return float(_DEFAULT_MPP_FALLBACK_FROM_OBJECTIVE_FACTOR / objective_power)

        raise WSIMetadataError(
            "Unable to determine MPP for slide "
            f"'{Path(path).expanduser().resolve()}'. "
            f"Missing metadata keys: {_PROP_MPP_X}, {_PROP_MPP_Y}, {_PROP_OBJECTIVE_POWER}."
        )

    def get_dimensions(self, path: str) -> tuple[int, int]:
        """Get level-0 slide dimensions.

        Args:
            path: Path to WSI file.

        Returns:
            Tuple ``(width, height)`` in level-0 pixels.

        Raises:
            WSIReaderError: If dimensions are invalid.
        """
        slide: Any = self.open(path)
        dims: Any = getattr(slide, "dimensions", None)
        if dims is None or len(dims) != 2:
            raise WSIReaderError(f"Slide '{path}' has invalid dimensions metadata: {dims!r}.")

        width: int = int(dims[0])
        height: int = int(dims[1])
        if width <= 0 or height <= 0:
            raise WSIReaderError(
                f"Slide '{path}' has non-positive dimensions: width={width}, height={height}."
            )
        return width, height

    # ------------------------------------------------------------------
    # Optional lifecycle helpers
    # ------------------------------------------------------------------
    def close(self, path: str) -> None:
        """Close a specific slide handle if present in cache."""
        normalized_path: str = str(Path(path).expanduser().resolve())
        with self._lock:
            slide: Optional[Any] = self._slides.pop(normalized_path, None)
            if slide is not None:
                try:
                    slide.close()
                except Exception:
                    # Best-effort close; no additional action required.
                    pass

    def close_all(self) -> None:
        """Close all cached slide handles."""
        with self._lock:
            for slide in list(self._slides.values()):
                try:
                    slide.close()
                except Exception:
                    pass
            self._slides.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_existing_path(path: str) -> str:
        """Resolve and validate an input path."""
        if not str(path).strip():
            raise ValueError("path must be a non-empty string.")
        resolved: Path = Path(path).expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"Slide path does not exist or is not a file: {resolved}")
        return str(resolved)

    @staticmethod
    def _find_float_property(properties: Dict[str, str], keys: Tuple[str, ...]) -> Optional[float]:
        """Return first parsable positive float property among candidate keys."""
        for key in keys:
            if key not in properties:
                continue
            raw_value: str = str(properties[key]).strip()
            if not raw_value:
                continue

            # Some vendors use "num/den" formats for resolutions.
            if "/" in raw_value:
                parts: Tuple[str, ...] = tuple(segment.strip() for segment in raw_value.split("/", maxsplit=1))
                if len(parts) == 2:
                    try:
                        numerator: float = float(parts[0])
                        denominator: float = float(parts[1])
                    except ValueError:
                        continue
                    if denominator != 0.0:
                        value: float = numerator / denominator
                        if value > 0.0:
                            return value
                continue

            try:
                value = float(raw_value)
            except ValueError:
                continue
            if value > 0.0:
                return value

        return None


__all__ = [
    "WSIReader",
    "WSIReaderError",
    "WSIBackendError",
    "WSIMetadataError",
]
