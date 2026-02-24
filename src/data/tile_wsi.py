"""WSI tiling utilities for TITAN data preprocessing.

This module implements deterministic, non-overlapping patch tiling over
segmented tissue masks and exposes the design-locked public interface:
- ``WSITiler.__init__(patch_size: int, stride: int)``
- ``WSITiler.tile_from_mask(wsi_path: str, mask: np.ndarray) -> list[tuple[int, int]]``
- ``WSITiler.filter_background(coords: list, min_tissue_ratio: float) -> list``

Coordinate convention:
- Returned coordinates are level-0 WSI pixel top-left anchors ``(x, y)``.
- Tiling uses 512x512 patches at 20x, consistent with the provided config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, List, Optional, Tuple

import numpy as np

from src.data.wsi_reader import WSIReader


# -----------------------------------------------------------------------------
# Paper/config-locked constants.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: Final[int] = 512
_MAGNIFICATION: Final[str] = "20x"
_FEATURE_DIM: Final[int] = 768

# Non-overlapping tiling policy for reproduction.
_STRIDE_PX: Final[int] = 512

# DataConfig default in this codebase for background filtering.
# Kept explicit here for deterministic, config-aligned behavior.
_DEFAULT_MIN_TISSUE_RATIO: Final[float] = 0.5


class WSITilerError(RuntimeError):
    """Base exception for WSI tiling failures."""


class WSITilerContextError(WSITilerError):
    """Raised when filter context is missing or invalid."""


class WSITiler:
    """Generate deterministic non-overlapping patch coordinates from tissue masks.

    Args:
        patch_size: Patch edge length in level-0 pixels. Must be 512.
        stride: Tiling stride in level-0 pixels. Must equal patch_size (512).
    """

    def __init__(self, patch_size: int = _PATCH_SIZE_PX, stride: int = _STRIDE_PX) -> None:
        if isinstance(patch_size, bool) or not isinstance(patch_size, (int, np.integer)):
            raise TypeError(f"patch_size must be an integer, got {type(patch_size).__name__}.")
        if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)):
            raise TypeError(f"stride must be an integer, got {type(stride).__name__}.")

        patch_size_int: int = int(patch_size)
        stride_int: int = int(stride)

        if patch_size_int <= 0:
            raise ValueError(f"patch_size must be > 0, got {patch_size_int}.")
        if stride_int <= 0:
            raise ValueError(f"stride must be > 0, got {stride_int}.")

        if patch_size_int != _PATCH_SIZE_PX:
            raise ValueError(
                f"patch_size must be {_PATCH_SIZE_PX} for TITAN reproduction, got {patch_size_int}."
            )
        if stride_int != patch_size_int:
            raise ValueError(
                f"stride must equal patch_size for non-overlapping tiling, got stride={stride_int}, "
                f"patch_size={patch_size_int}."
            )

        self.patch_size: int = patch_size_int
        self.stride: int = stride_int
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM

        self._reader: WSIReader = WSIReader(backend="openslide")

        # Default filtering threshold used by tile_from_mask.
        self._default_min_tissue_ratio: float = _DEFAULT_MIN_TISSUE_RATIO

        # Internal context for filter_background().
        self._context_ready: bool = False
        self._mask_binary: Optional[np.ndarray] = None
        self._integral_mask: Optional[np.ndarray] = None
        self._wsi_width: int = 0
        self._wsi_height: int = 0
        self._mask_width: int = 0
        self._mask_height: int = 0
        self._scale_x: float = 0.0
        self._scale_y: float = 0.0

    def tile_from_mask(self, wsi_path: str, mask: np.ndarray) -> list[tuple[int, int]]:
        """Generate filtered tile coordinates from a slide path and tissue mask.

        Args:
            wsi_path: Path to a whole-slide image.
            mask: Binary tissue mask in thumbnail space (H, W).

        Returns:
            Deterministic row-major list of level-0 coordinates ``[(x, y), ...]``.
        """
        normalized_path: str = self._validate_wsi_path(wsi_path)
        binary_mask: np.ndarray = self._validate_and_binarize_mask(mask)

        wsi_width, wsi_height = self._reader.get_dimensions(normalized_path)
        if wsi_width <= 0 or wsi_height <= 0:
            raise WSITilerError(
                f"Invalid slide dimensions for '{normalized_path}': width={wsi_width}, height={wsi_height}."
            )

        if self.patch_size > wsi_width or self.patch_size > wsi_height:
            self._set_filter_context(binary_mask=binary_mask, wsi_width=wsi_width, wsi_height=wsi_height)
            return []

        x_anchors: np.ndarray = np.arange(
            0,
            wsi_width - self.patch_size + 1,
            self.stride,
            dtype=np.int64,
        )
        y_anchors: np.ndarray = np.arange(
            0,
            wsi_height - self.patch_size + 1,
            self.stride,
            dtype=np.int64,
        )

        if x_anchors.size == 0 or y_anchors.size == 0:
            self._set_filter_context(binary_mask=binary_mask, wsi_width=wsi_width, wsi_height=wsi_height)
            return []

        grid_x, grid_y = np.meshgrid(x_anchors, y_anchors, indexing="xy")
        candidate_coords: List[Tuple[int, int]] = [
            (int(x), int(y)) for x, y in zip(grid_x.ravel(order="C"), grid_y.ravel(order="C"))
        ]

        self._set_filter_context(binary_mask=binary_mask, wsi_width=wsi_width, wsi_height=wsi_height)

        filtered_coords: List[Tuple[int, int]] = self.filter_background(
            coords=candidate_coords,
            min_tissue_ratio=self._default_min_tissue_ratio,
        )

        return filtered_coords

    def filter_background(
        self,
        coords: list,
        min_tissue_ratio: float = _DEFAULT_MIN_TISSUE_RATIO,
    ) -> list:
        """Filter coordinates by tissue occupancy in the active mask context.

        Args:
            coords: Candidate list of ``(x, y)`` level-0 top-left coordinates.
            min_tissue_ratio: Minimum tissue fraction in [0, 1].

        Returns:
            Filtered list preserving the input order.
        """
        if not isinstance(coords, list):
            raise TypeError(f"coords must be a list, got {type(coords).__name__}.")

        threshold: float = float(min_tissue_ratio)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"min_tissue_ratio must be within [0, 1], got {threshold}.")

        self._ensure_filter_context()

        if len(coords) == 0:
            return []

        kept_coords: List[Tuple[int, int]] = []
        for item in coords:
            x, y = self._validate_coord(item)

            if x < 0 or y < 0:
                continue
            if x + self.patch_size > self._wsi_width:
                continue
            if y + self.patch_size > self._wsi_height:
                continue

            tissue_ratio: float = self._tissue_ratio_for_patch(x=x, y=y)
            if tissue_ratio >= threshold:
                kept_coords.append((x, y))

        return kept_coords

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_wsi_path(wsi_path: str) -> str:
        if not isinstance(wsi_path, str):
            raise TypeError(f"wsi_path must be str, got {type(wsi_path).__name__}.")
        if not wsi_path.strip():
            raise ValueError("wsi_path cannot be empty.")
        path_obj: Path = Path(wsi_path).expanduser().resolve()
        if not path_obj.exists() or not path_obj.is_file():
            raise FileNotFoundError(f"WSI file not found: {path_obj}")
        return str(path_obj)

    @staticmethod
    def _validate_and_binarize_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask must be np.ndarray, got {type(mask).__name__}.")
        if mask.ndim != 2:
            raise ValueError(f"mask must be rank-2 (H, W), got shape={mask.shape}.")
        if mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError(f"mask must have positive shape, got {mask.shape}.")

        if mask.dtype == np.bool_:
            binary: np.ndarray = mask.astype(np.uint8)
            return np.ascontiguousarray(binary)

        if np.issubdtype(mask.dtype, np.integer):
            binary = (mask > 0).astype(np.uint8)
            return np.ascontiguousarray(binary)

        if np.issubdtype(mask.dtype, np.floating):
            finite_mask: np.ndarray = np.nan_to_num(mask, nan=0.0, posinf=1.0, neginf=0.0)
            binary = (finite_mask > 0.0).astype(np.uint8)
            return np.ascontiguousarray(binary)

        raise TypeError(f"Unsupported mask dtype: {mask.dtype}.")

    def _set_filter_context(self, binary_mask: np.ndarray, wsi_width: int, wsi_height: int) -> None:
        self._mask_binary = np.ascontiguousarray(binary_mask.astype(np.uint8))
        self._integral_mask = self._build_integral_image(self._mask_binary)

        self._mask_height = int(self._mask_binary.shape[0])
        self._mask_width = int(self._mask_binary.shape[1])
        self._wsi_width = int(wsi_width)
        self._wsi_height = int(wsi_height)

        if self._wsi_width <= 0 or self._wsi_height <= 0:
            raise WSITilerContextError(
                f"Invalid WSI dimensions in context: width={self._wsi_width}, height={self._wsi_height}."
            )
        if self._mask_width <= 0 or self._mask_height <= 0:
            raise WSITilerContextError(
                f"Invalid mask dimensions in context: width={self._mask_width}, height={self._mask_height}."
            )

        self._scale_x = float(self._mask_width) / float(self._wsi_width)
        self._scale_y = float(self._mask_height) / float(self._wsi_height)

        if self._scale_x <= 0.0 or self._scale_y <= 0.0:
            raise WSITilerContextError(
                f"Non-positive coordinate scale factors: scale_x={self._scale_x}, scale_y={self._scale_y}."
            )

        self._context_ready = True

    @staticmethod
    def _build_integral_image(binary_mask: np.ndarray) -> np.ndarray:
        # Integral image with 1-pixel zero padding for O(1) rectangle sum queries.
        # Shape: (H + 1, W + 1)
        mask_i64: np.ndarray = binary_mask.astype(np.int64, copy=False)
        integral: np.ndarray = np.zeros(
            (mask_i64.shape[0] + 1, mask_i64.shape[1] + 1),
            dtype=np.int64,
        )
        integral[1:, 1:] = mask_i64.cumsum(axis=0).cumsum(axis=1)
        return integral

    def _ensure_filter_context(self) -> None:
        if not self._context_ready:
            raise WSITilerContextError(
                "filter_background requires an active context. Call tile_from_mask(...) first."
            )
        if self._mask_binary is None or self._integral_mask is None:
            raise WSITilerContextError("Filter context is incomplete (missing mask or integral image).")

    @staticmethod
    def _validate_coord(item: object) -> Tuple[int, int]:
        if not isinstance(item, (tuple, list)):
            raise TypeError(f"Each coordinate must be tuple/list, got {type(item).__name__}.")
        if len(item) != 2:
            raise ValueError(f"Each coordinate must have length 2, got {len(item)}.")

        x_raw: object = item[0]
        y_raw: object = item[1]

        if isinstance(x_raw, bool) or not isinstance(x_raw, (int, np.integer)):
            raise TypeError(f"Coordinate x must be integer, got {type(x_raw).__name__}.")
        if isinstance(y_raw, bool) or not isinstance(y_raw, (int, np.integer)):
            raise TypeError(f"Coordinate y must be integer, got {type(y_raw).__name__}.")

        return int(x_raw), int(y_raw)

    def _tissue_ratio_for_patch(self, x: int, y: int) -> float:
        # Map level-0 patch bounds to mask-space bounds.
        x0_m: int = int(np.floor(float(x) * self._scale_x))
        y0_m: int = int(np.floor(float(y) * self._scale_y))
        x1_m: int = int(np.ceil(float(x + self.patch_size) * self._scale_x))
        y1_m: int = int(np.ceil(float(y + self.patch_size) * self._scale_y))

        x0_m = int(np.clip(x0_m, 0, self._mask_width))
        y0_m = int(np.clip(y0_m, 0, self._mask_height))
        x1_m = int(np.clip(x1_m, 0, self._mask_width))
        y1_m = int(np.clip(y1_m, 0, self._mask_height))

        if x1_m <= x0_m:
            x1_m = min(self._mask_width, x0_m + 1)
        if y1_m <= y0_m:
            y1_m = min(self._mask_height, y0_m + 1)

        if x1_m <= x0_m or y1_m <= y0_m:
            return 0.0

        assert self._integral_mask is not None
        tissue_pixels: int = int(
            self._integral_mask[y1_m, x1_m]
            - self._integral_mask[y0_m, x1_m]
            - self._integral_mask[y1_m, x0_m]
            + self._integral_mask[y0_m, x0_m]
        )
        region_area: int = int((y1_m - y0_m) * (x1_m - x0_m))

        if region_area <= 0:
            return 0.0

        ratio: float = float(tissue_pixels) / float(region_area)
        if ratio < 0.0:
            return 0.0
        if ratio > 1.0:
            return 1.0
        return ratio


__all__ = [
    "WSITiler",
    "WSITilerError",
    "WSITilerContextError",
]
