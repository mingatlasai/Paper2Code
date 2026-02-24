"""Tissue segmentation utilities for WSI preprocessing.

This module implements CLAM-style tissue segmentation used in TITAN data
preparation:
1) HSV saturation thresholding on a low-resolution thumbnail,
2) median blur denoising,
3) morphological closing,
4) contour extraction with minimum-area filtering.

Public interface follows the design contract exactly:
- ``TissueSegmenter.__init__(sat_thresh: float, min_area: int)``
- ``TissueSegmenter.segment_thumbnail(wsi_thumb: np.ndarray) -> np.ndarray``
- ``TissueSegmenter.postprocess_mask(mask: np.ndarray) -> np.ndarray``
- ``TissueSegmenter.extract_contours(mask: np.ndarray) -> list``
"""

from __future__ import annotations

from typing import Final, List, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover - import-time environment dependent
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR: Exception | None = exc
else:
    _CV2_IMPORT_ERROR = None


# -----------------------------------------------------------------------------
# Defaults aligned with project configuration contract.
# -----------------------------------------------------------------------------
_DEFAULT_SATURATION_THRESHOLD: Final[float] = 8.0
_DEFAULT_MIN_CONTOUR_AREA: Final[int] = 256
_DEFAULT_MEDIAN_BLUR_KSIZE: Final[int] = 7
_DEFAULT_MORPH_CLOSE_KSIZE: Final[int] = 7
_DEFAULT_MORPH_CLOSE_ITERATIONS: Final[int] = 1

_BINARY_OFF: Final[int] = 0
_BINARY_ON: Final[int] = 255


class TissueSegmenterError(RuntimeError):
    """Base exception for segmentation failures."""


class TissueSegmenter:
    """Deterministic thumbnail-space tissue segmenter.

    Args:
        sat_thresh: HSV saturation threshold in ``[0, 255]``.
        min_area: Minimum contour area in thumbnail pixels.
    """

    def __init__(
        self,
        sat_thresh: float = _DEFAULT_SATURATION_THRESHOLD,
        min_area: int = _DEFAULT_MIN_CONTOUR_AREA,
    ) -> None:
        self._ensure_cv2_available()

        if isinstance(sat_thresh, bool):
            raise ValueError("sat_thresh cannot be bool.")
        if not isinstance(sat_thresh, (int, float, np.integer, np.floating)):
            raise TypeError(f"sat_thresh must be numeric, got {type(sat_thresh).__name__}.")

        sat_thresh_float: float = float(sat_thresh)
        if sat_thresh_float < 0.0 or sat_thresh_float > 255.0:
            raise ValueError(f"sat_thresh must be within [0, 255], got {sat_thresh_float}.")

        if isinstance(min_area, bool):
            raise ValueError("min_area cannot be bool.")
        if not isinstance(min_area, (int, np.integer)):
            raise TypeError(f"min_area must be integer, got {type(min_area).__name__}.")

        min_area_int: int = int(min_area)
        if min_area_int <= 0:
            raise ValueError(f"min_area must be > 0, got {min_area_int}.")

        self.sat_thresh: float = sat_thresh_float
        self.min_area: int = min_area_int

        self._median_blur_ksize: int = _DEFAULT_MEDIAN_BLUR_KSIZE
        self._morph_close_ksize: int = _DEFAULT_MORPH_CLOSE_KSIZE
        self._morph_close_iterations: int = _DEFAULT_MORPH_CLOSE_ITERATIONS

    def segment_thumbnail(self, wsi_thumb: np.ndarray) -> np.ndarray:
        """Segment tissue by thresholding HSV saturation.

        Args:
            wsi_thumb: Thumbnail image in RGB-like layout, shape ``(H, W, C)``.

        Returns:
            A binary ``uint8`` mask of shape ``(H, W)`` with values in ``{0, 255}``.
        """
        rgb_thumb: np.ndarray = self._validate_and_prepare_thumbnail(wsi_thumb)

        hsv_thumb: np.ndarray = cv2.cvtColor(rgb_thumb, cv2.COLOR_RGB2HSV)
        saturation_channel: np.ndarray = hsv_thumb[..., 1]

        _, binary_mask = cv2.threshold(
            saturation_channel,
            float(self.sat_thresh),
            float(_BINARY_ON),
            cv2.THRESH_BINARY,
        )

        return self._as_binary_mask(binary_mask)

    def postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Denoise and fill tissue mask via median blur and morphological closing.

        Args:
            mask: Input 2D binary-like mask.

        Returns:
            Postprocessed binary ``uint8`` mask with same shape as input.
        """
        binary_mask: np.ndarray = self._validate_and_prepare_mask(mask)

        # Median blur suppresses isolated salt-and-pepper artifacts.
        blurred_mask: np.ndarray = cv2.medianBlur(binary_mask, self._median_blur_ksize)

        close_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self._morph_close_ksize, self._morph_close_ksize),
        )
        closed_mask: np.ndarray = cv2.morphologyEx(
            blurred_mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=self._morph_close_iterations,
        )

        if closed_mask.shape != binary_mask.shape:
            raise TissueSegmenterError(
                "Postprocessing changed mask shape unexpectedly: "
                f"input={binary_mask.shape}, output={closed_mask.shape}."
            )

        return self._as_binary_mask(closed_mask)

    def extract_contours(self, mask: np.ndarray) -> list:
        """Extract external tissue contours and filter by minimum area.

        Args:
            mask: Input 2D binary-like mask.

        Returns:
            A deterministic list of contours (OpenCV contour arrays).
        """
        binary_mask: np.ndarray = self._validate_and_prepare_mask(mask)

        find_result = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if len(find_result) == 3:
            _, contours, _ = find_result
        elif len(find_result) == 2:
            contours, _ = find_result
        else:  # pragma: no cover - defensive guard for unexpected OpenCV API changes
            raise TissueSegmenterError("Unexpected return structure from cv2.findContours.")

        valid_contours: List[np.ndarray] = []
        for contour in contours:
            if contour is None or len(contour) == 0:
                continue
            area: float = float(cv2.contourArea(contour))
            if area >= float(self.min_area):
                valid_contours.append(contour)

        # Ensure deterministic ordering across runs/processes.
        valid_contours.sort(key=self._contour_sort_key)

        return list(valid_contours)

    @staticmethod
    def _ensure_cv2_available() -> None:
        if cv2 is None:
            detail: str = f" ({_CV2_IMPORT_ERROR})" if _CV2_IMPORT_ERROR is not None else ""
            raise TissueSegmenterError(
                "OpenCV is required for tissue segmentation. "
                "Install opencv-python==4.9.0.80."
                f"{detail}"
            )

    @staticmethod
    def _validate_and_prepare_thumbnail(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise TypeError(f"wsi_thumb must be np.ndarray, got {type(image).__name__}.")
        if image.ndim != 3:
            raise ValueError(f"wsi_thumb must be rank-3 (H, W, C), got shape={image.shape}.")
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"wsi_thumb must have positive spatial dimensions, got shape={image.shape}.")
        if image.shape[2] not in (3, 4):
            raise ValueError(
                "wsi_thumb must have 3 (RGB) or 4 (RGBA) channels, "
                f"got shape={image.shape}."
            )

        image_u8: np.ndarray = TissueSegmenter._coerce_to_uint8(image)
        if image_u8.shape[2] == 4:
            image_u8 = image_u8[:, :, :3]
        return image_u8

    @staticmethod
    def _validate_and_prepare_mask(mask: np.ndarray) -> np.ndarray:
        if not isinstance(mask, np.ndarray):
            raise TypeError(f"mask must be np.ndarray, got {type(mask).__name__}.")
        if mask.ndim != 2:
            raise ValueError(f"mask must be rank-2 (H, W), got shape={mask.shape}.")
        if mask.shape[0] <= 0 or mask.shape[1] <= 0:
            raise ValueError(f"mask must have positive spatial dimensions, got shape={mask.shape}.")

        return TissueSegmenter._as_binary_mask(mask)

    @staticmethod
    def _coerce_to_uint8(array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8:
            return np.ascontiguousarray(array)

        if np.issubdtype(array.dtype, np.bool_):
            return np.ascontiguousarray(array.astype(np.uint8) * _BINARY_ON)

        if np.issubdtype(array.dtype, np.integer):
            clipped: np.ndarray = np.clip(array, _BINARY_OFF, _BINARY_ON)
            return np.ascontiguousarray(clipped.astype(np.uint8))

        if np.issubdtype(array.dtype, np.floating):
            finite_values: np.ndarray = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            # If values appear normalized, scale to 0..255.
            if float(np.max(finite_values)) <= 1.0:
                finite_values = finite_values * float(_BINARY_ON)
            clipped_f: np.ndarray = np.clip(finite_values, 0.0, 255.0)
            return np.ascontiguousarray(clipped_f.astype(np.uint8))

        raise TypeError(f"Unsupported dtype for conversion to uint8: {array.dtype}.")

    @staticmethod
    def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
        mask_u8: np.ndarray = TissueSegmenter._coerce_to_uint8(mask)
        binary_mask: np.ndarray = np.where(mask_u8 > _BINARY_OFF, _BINARY_ON, _BINARY_OFF).astype(np.uint8)
        return np.ascontiguousarray(binary_mask)

    @staticmethod
    def _contour_sort_key(contour: np.ndarray) -> Tuple[int, int, int, int]:
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = cv2.boundingRect(contour)
        area: int = int(round(float(cv2.contourArea(contour))))
        # Row-major spatial order first, then larger contours first for tie-break stability.
        return (int(y), int(x), -area, int(w * h))


__all__ = [
    "TissueSegmenterError",
    "TissueSegmenter",
]
