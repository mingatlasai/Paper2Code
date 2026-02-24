"""Patch encoder interfaces and CONCHV1.5 adapter.

This module implements the design-locked contracts:
- ``PatchEncoder.encode(patches: object) -> object``
- ``PatchEncoder.feature_dim() -> int``
- ``ConchPatchEncoder.__init__(model_name: str, device: str, precision: str) -> None``
- ``ConchPatchEncoder.preprocess(patches: object) -> object``
- ``ConchPatchEncoder.encode(patches: object) -> object``
- ``ConchPatchEncoder.feature_dim() -> int``

Paper/config constraints enforced here:
- Patch input geometry from preprocessing: 512x512 at 20x.
- Patch encoder input resize: 512 -> 448.
- Normalization: ImageNet mean/std.
- THREADS-compatible patch feature width: 768.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as torch_functional

try:
    import timm
except Exception:  # pragma: no cover - optional import at runtime.
    timm = None  # type: ignore[assignment]

try:
    import open_clip
except Exception:  # pragma: no cover - optional import at runtime.
    open_clip = None  # type: ignore[assignment]


LOGGER: logging.Logger = logging.getLogger(__name__)

# Config-anchored defaults.
DEFAULT_MODEL_NAME: str = "CONCHV1.5"
DEFAULT_DEVICE: str = "cpu"
DEFAULT_PRECISION: str = "fp32"

DEFAULT_PATCH_SIZE: int = 512
DEFAULT_INPUT_RESIZE: int = 448
DEFAULT_TARGET_MAGNIFICATION: int = 20
DEFAULT_FEATURE_DIM: int = 768
DEFAULT_MIN_PATCH_COUNT: int = 1
DEFAULT_MAX_BATCH_SIZE: int = 256

IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

_ALLOWED_PRECISIONS: Tuple[str, ...] = ("fp32", "fp16", "bf16")


class PatchEncoderError(Exception):
    """Base exception for patch encoder failures."""


class PatchEncoderConfigError(PatchEncoderError):
    """Raised when patch encoder configuration is invalid."""


class PatchEncoderInputError(PatchEncoderError):
    """Raised when patch input payload is malformed."""


class PatchEncoderRuntimeError(PatchEncoderError):
    """Raised when backend inference/model loading fails."""


class PatchEncoder(ABC):
    """Abstract patch encoder contract."""

    @abstractmethod
    def encode(self, patches: object) -> object:
        """Encode patches into patch-level feature vectors."""

    @abstractmethod
    def feature_dim(self) -> int:
        """Return the encoder output feature width."""


class ConchPatchEncoder(PatchEncoder):
    """CONCHV1.5-compatible patch encoder adapter.

    Notes:
    - Uses deterministic preprocessing (resize to 448, ImageNet normalization).
    - Tries OpenCLIP backend if explicitly enabled via env var
      ``THREADS_USE_OPEN_CLIP=1`` and available.
    - Falls back to local TIMM ViT-L/16 model without external downloads.
    - Always returns THREADS-compatible 768-d vectors.
    """

    def __init__(self, model_name: str, device: str, precision: str) -> None:
        self._model_name: str = self._normalize_model_name(model_name)
        self._device: str = self._normalize_device(device)
        self._precision: str = self._normalize_precision(precision, self._device)
        self._dtype: torch.dtype = self._precision_to_dtype(self._precision)

        self._input_resize: int = DEFAULT_INPUT_RESIZE
        self._source_patch_size: int = DEFAULT_PATCH_SIZE
        self._target_magnification: int = DEFAULT_TARGET_MAGNIFICATION
        self._feature_width: int = DEFAULT_FEATURE_DIM
        self._max_batch_size: int = DEFAULT_MAX_BATCH_SIZE

        self._model: Optional[torch.nn.Module] = None
        self._backend_name: str = ""

        self._mean_tensor: torch.Tensor = torch.tensor(
            IMAGENET_MEAN, dtype=torch.float32
        ).view(1, 3, 1, 1)
        self._std_tensor: torch.Tensor = torch.tensor(
            IMAGENET_STD, dtype=torch.float32
        ).view(1, 3, 1, 1)

    def preprocess(self, patches: object) -> object:
        """Convert raw patch payload to model-ready tensor [N, 3, 448, 448]."""
        patch_tensor: torch.Tensor = self._to_nchw_float_tensor(patches)
        if patch_tensor.shape[0] < DEFAULT_MIN_PATCH_COUNT:
            raise PatchEncoderInputError("preprocess received no patches.")

        # Resize 512 -> 448 per config/paper.
        resized_tensor: torch.Tensor = torch_functional.interpolate(
            patch_tensor,
            size=(self._input_resize, self._input_resize),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

        normalized_tensor: torch.Tensor = (
            resized_tensor - self._mean_tensor
        ) / self._std_tensor
        return normalized_tensor.contiguous()

    def encode(self, patches: object) -> object:
        """Encode patch batch into [N, 768] features.

        Accepts either:
        - raw patches (list[bytes], list[np.ndarray], np.ndarray, torch.Tensor)
        - preprocessed tensors [N, 3, 448, 448]
        """
        model_input: torch.Tensor = self._ensure_model_input_tensor(patches)
        if model_input.shape[0] < DEFAULT_MIN_PATCH_COUNT:
            raise PatchEncoderInputError("encode received no patches.")

        self._ensure_model_loaded()
        if self._model is None:
            raise PatchEncoderRuntimeError("Internal model is not initialized.")

        model_input = model_input.to(self._device, non_blocking=True)
        if self._dtype in {torch.float16, torch.bfloat16}:
            model_input = model_input.to(self._dtype)

        output_chunks: List[torch.Tensor] = []
        with torch.inference_mode():
            start_index: int = 0
            while start_index < model_input.shape[0]:
                end_index: int = min(
                    start_index + self._max_batch_size, model_input.shape[0]
                )
                batch_tensor: torch.Tensor = model_input[start_index:end_index]
                feature_tensor: torch.Tensor = self._forward_features(batch_tensor)
                feature_tensor = self._coerce_feature_width(
                    feature_tensor, self._feature_width
                )
                output_chunks.append(feature_tensor)
                start_index = end_index

        if len(output_chunks) == 1:
            output_tensor: torch.Tensor = output_chunks[0]
        else:
            output_tensor = torch.cat(output_chunks, dim=0)

        if output_tensor.shape[1] != self._feature_width:
            raise PatchEncoderRuntimeError(
                f"Feature width mismatch after encoding: expected {self._feature_width}, "
                f"got {output_tensor.shape[1]}."
            )

        return output_tensor.detach().to(torch.float32).cpu().contiguous()

    def feature_dim(self) -> int:
        """Return patch feature width expected by THREADS slide encoder."""
        return self._feature_width

    def _normalize_model_name(self, model_name: str) -> str:
        normalized_name: str = str(model_name).strip()
        if normalized_name == "":
            normalized_name = DEFAULT_MODEL_NAME

        lowered_name: str = normalized_name.lower().replace(" ", "")
        valid_aliases: Tuple[str, ...] = (
            "conchv1.5",
            "conch_v1.5",
            "conchv15",
            "conch",
        )
        if lowered_name not in valid_aliases:
            raise PatchEncoderConfigError(
                "ConchPatchEncoder only supports CONCHV1.5-compatible names. "
                f"Got model_name={model_name!r}."
            )
        return normalized_name

    def _normalize_device(self, device: str) -> str:
        normalized_device: str = (
            str(device).strip().lower() if device is not None else ""
        )
        if normalized_device == "":
            normalized_device = DEFAULT_DEVICE

        if normalized_device.startswith("cuda"):
            if not torch.cuda.is_available():
                LOGGER.warning(
                    "CUDA device requested but unavailable; falling back to CPU."
                )
                return "cpu"
            return normalized_device

        if normalized_device == "cpu":
            return "cpu"

        raise PatchEncoderConfigError(
            f"Unsupported device={device!r}. Expected 'cpu' or 'cuda[:index]'."
        )

    def _normalize_precision(self, precision: str, device: str) -> str:
        normalized_precision: str = (
            str(precision).strip().lower() if precision is not None else ""
        )
        if normalized_precision == "":
            normalized_precision = DEFAULT_PRECISION

        if normalized_precision not in _ALLOWED_PRECISIONS:
            raise PatchEncoderConfigError(
                f"Unsupported precision={precision!r}. "
                f"Expected one of {_ALLOWED_PRECISIONS}."
            )

        # Keep CPU path robust.
        if device == "cpu" and normalized_precision in {"fp16", "bf16"}:
            LOGGER.warning(
                "precision=%s requested on CPU; using fp32 for stable inference.",
                normalized_precision,
            )
            return "fp32"

        return normalized_precision

    def _precision_to_dtype(self, precision: str) -> torch.dtype:
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        return torch.float32

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return

        backend_errors: List[str] = []

        # Optional OpenCLIP path gated by env to avoid accidental network fetch.
        use_open_clip_env: str = str(os.getenv("THREADS_USE_OPEN_CLIP", "0")).strip()
        should_try_open_clip: bool = use_open_clip_env == "1"
        if should_try_open_clip:
            try:
                self._model = self._build_open_clip_model()
                self._backend_name = "open_clip"
            except Exception as exc:  # noqa: BLE001
                backend_errors.append(f"open_clip: {exc}")
                self._model = None

        if self._model is None:
            try:
                self._model = self._build_timm_model()
                self._backend_name = "timm_vitl16_fallback"
            except Exception as exc:  # noqa: BLE001
                backend_errors.append(f"timm: {exc}")
                self._model = None

        if self._model is None:
            raise PatchEncoderRuntimeError(
                "Failed to initialize any patch-encoder backend. "
                + (
                    "; ".join(backend_errors)
                    if backend_errors
                    else "no backend error details"
                )
            )

        self._model.eval()
        self._model.to(self._device)
        LOGGER.info(
            "Initialized ConchPatchEncoder backend=%s device=%s precision=%s "
            "feature_dim=%d",
            self._backend_name,
            self._device,
            self._precision,
            self._feature_width,
        )

    def _build_open_clip_model(self) -> torch.nn.Module:
        if open_clip is None:
            raise PatchEncoderRuntimeError("open_clip is not available.")

        # Known likely names; no external download is forced here.
        candidate_names: Tuple[str, ...] = (
            self._model_name,
            "conch_v1_5",
            "conch_v1.5",
            "hf-hub:MahmoodLab/conch-v1-5",
            "hf-hub:MahmoodLab/conch-v1_5",
        )

        last_error: Optional[Exception] = None
        for candidate_name in candidate_names:
            try:
                model: torch.nn.Module = open_clip.create_model(
                    model_name=candidate_name,
                    pretrained=None,
                    device=self._device,
                )
                return model
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        raise PatchEncoderRuntimeError(
            "Unable to construct OpenCLIP CONCH model from known aliases. "
            f"Last error: {last_error}"
        )

    def _build_timm_model(self) -> torch.nn.Module:
        if timm is None:
            raise PatchEncoderRuntimeError("timm is not available.")

        # Deterministic local fallback with 768-d ViT embeddings.
        model: torch.nn.Module = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
        return model

    def _ensure_model_input_tensor(self, patches: object) -> torch.Tensor:
        # Already preprocessed [N, 3, 448, 448]
        if isinstance(patches, torch.Tensor):
            tensor_value: torch.Tensor = patches.detach()
            if (
                tensor_value.ndim == 4
                and tensor_value.shape[1] == 3
                and tensor_value.shape[2] == self._input_resize
                and tensor_value.shape[3] == self._input_resize
            ):
                return tensor_value.to(torch.float32).cpu().contiguous()

        # Otherwise treat as raw patches and preprocess.
        preprocessed: object = self.preprocess(patches)
        if not isinstance(preprocessed, torch.Tensor):
            raise PatchEncoderInputError("preprocess must return a torch.Tensor.")
        return preprocessed

    def _to_nchw_float_tensor(self, patches: object) -> torch.Tensor:
        if isinstance(patches, torch.Tensor):
            return self._tensor_to_nchw_float_tensor(patches)

        if isinstance(patches, np.ndarray):
            return self._ndarray_to_nchw_float_tensor(patches)

        if isinstance(patches, (list, tuple)):
            return self._sequence_to_nchw_float_tensor(patches)

        raise PatchEncoderInputError(
            f"Unsupported patch input type: {type(patches).__name__}."
        )

    def _tensor_to_nchw_float_tensor(self, tensor_value: torch.Tensor) -> torch.Tensor:
        if tensor_value.ndim == 4:
            # NCHW
            if tensor_value.shape[1] == 3:
                output_tensor: torch.Tensor = tensor_value.to(torch.float32)
            # NHWC
            elif tensor_value.shape[3] == 3:
                output_tensor = tensor_value.permute(0, 3, 1, 2).to(torch.float32)
            else:
                raise PatchEncoderInputError(
                    "4D tensor must be NCHW or NHWC with 3 channels, "
                    f"got shape={tuple(tensor_value.shape)}."
                )
        elif tensor_value.ndim == 3:
            # CHW
            if tensor_value.shape[0] == 3:
                output_tensor = tensor_value.unsqueeze(0).to(torch.float32)
            # HWC
            elif tensor_value.shape[2] == 3:
                output_tensor = (
                    tensor_value.permute(2, 0, 1).unsqueeze(0).to(torch.float32)
                )
            else:
                raise PatchEncoderInputError(
                    "3D tensor must be CHW or HWC with 3 channels, "
                    f"got shape={tuple(tensor_value.shape)}."
                )
        else:
            raise PatchEncoderInputError(
                "Tensor patches must be rank-3 or rank-4, "
                f"got rank={tensor_value.ndim}."
            )

        output_tensor = output_tensor.detach().cpu().contiguous()
        output_tensor = self._normalize_pixel_range(output_tensor)
        self._validate_source_patch_shape(output_tensor)
        return output_tensor

    def _ndarray_to_nchw_float_tensor(self, array_value: np.ndarray) -> torch.Tensor:
        if array_value.ndim == 4:
            if array_value.shape[1] == 3:
                output_array: np.ndarray = array_value
            elif array_value.shape[3] == 3:
                output_array = np.transpose(array_value, (0, 3, 1, 2))
            else:
                raise PatchEncoderInputError(
                    "4D ndarray must be NCHW or NHWC with 3 channels, "
                    f"got shape={array_value.shape}."
                )
        elif array_value.ndim == 3:
            if array_value.shape[0] == 3:
                output_array = np.expand_dims(array_value, axis=0)
            elif array_value.shape[2] == 3:
                output_array = np.expand_dims(
                    np.transpose(array_value, (2, 0, 1)), axis=0
                )
            else:
                raise PatchEncoderInputError(
                    "3D ndarray must be CHW or HWC with 3 channels, "
                    f"got shape={array_value.shape}."
                )
        else:
            raise PatchEncoderInputError(
                "ndarray patches must be rank-3 or rank-4, "
                f"got rank={array_value.ndim}."
            )

        output_tensor: torch.Tensor = torch.from_numpy(
            np.ascontiguousarray(output_array)
        ).to(torch.float32)
        output_tensor = self._normalize_pixel_range(output_tensor)
        self._validate_source_patch_shape(output_tensor)
        return output_tensor

    def _sequence_to_nchw_float_tensor(
        self, sequence_value: Sequence[object]
    ) -> torch.Tensor:
        if len(sequence_value) < DEFAULT_MIN_PATCH_COUNT:
            raise PatchEncoderInputError("Patch sequence is empty.")

        tensor_list: List[torch.Tensor] = []
        for index, item in enumerate(sequence_value):
            tensor_list.append(self._single_patch_to_chw_tensor(item, index))

        stacked: torch.Tensor = torch.stack(tensor_list, dim=0).to(torch.float32)
        stacked = self._normalize_pixel_range(stacked)
        self._validate_source_patch_shape(stacked)
        return stacked

    def _single_patch_to_chw_tensor(self, item: object, index: int) -> torch.Tensor:
        if isinstance(item, bytes):
            expected_length: int = self._source_patch_size * self._source_patch_size * 3
            if len(item) != expected_length:
                raise PatchEncoderInputError(
                    f"Patch bytes at index={index} have invalid size {len(item)}; "
                    f"expected {expected_length}."
                )
            array_value: np.ndarray = np.frombuffer(item, dtype=np.uint8).reshape(
                self._source_patch_size,
                self._source_patch_size,
                3,
            )
            chw_array: np.ndarray = np.transpose(array_value, (2, 0, 1))
            return torch.from_numpy(np.ascontiguousarray(chw_array))

        if isinstance(item, np.ndarray):
            array_value = item
            if array_value.ndim == 3 and array_value.shape[2] == 3:
                chw_array = np.transpose(array_value, (2, 0, 1))
            elif array_value.ndim == 3 and array_value.shape[0] == 3:
                chw_array = array_value
            else:
                raise PatchEncoderInputError(
                    "Patch ndarray at index="
                    f"{index} must be HWC/CHW with 3 channels; got shape={array_value.shape}."
                )
            return torch.from_numpy(np.ascontiguousarray(chw_array))

        if isinstance(item, torch.Tensor):
            tensor_value: torch.Tensor = item.detach().cpu()
            if tensor_value.ndim == 3 and tensor_value.shape[0] == 3:
                return tensor_value.contiguous()
            if tensor_value.ndim == 3 and tensor_value.shape[2] == 3:
                return tensor_value.permute(2, 0, 1).contiguous()
            raise PatchEncoderInputError(
                "Patch tensor at index="
                f"{index} must be HWC/CHW with 3 channels; got shape={tuple(tensor_value.shape)}."
            )

        raise PatchEncoderInputError(
            f"Unsupported patch type at index={index}: {type(item).__name__}."
        )

    def _normalize_pixel_range(self, patch_tensor: torch.Tensor) -> torch.Tensor:
        if patch_tensor.numel() == 0:
            return patch_tensor

        if torch.max(patch_tensor) > 1.0 or torch.min(patch_tensor) < 0.0:
            patch_tensor = patch_tensor / 255.0

        patch_tensor = torch.clamp(patch_tensor, 0.0, 1.0)
        return patch_tensor

    def _validate_source_patch_shape(self, patch_tensor: torch.Tensor) -> None:
        if patch_tensor.ndim != 4:
            raise PatchEncoderInputError(
                "Patch tensor must be rank-4 after normalization, "
                f"got rank={patch_tensor.ndim}."
            )

        channels: int = int(patch_tensor.shape[1])
        height: int = int(patch_tensor.shape[2])
        width: int = int(patch_tensor.shape[3])

        if channels != 3:
            raise PatchEncoderInputError(
                f"Patch tensor must have 3 channels, got {channels}."
            )

        if height != self._source_patch_size or width != self._source_patch_size:
            raise PatchEncoderInputError(
                "Patch tensor must match source geometry before resize: "
                f"expected ({self._source_patch_size}, {self._source_patch_size}), "
                f"got ({height}, {width})."
            )

    def _forward_features(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise PatchEncoderRuntimeError("Model is not initialized.")

        # OpenCLIP path.
        if self._backend_name == "open_clip":
            model_value: Any = self._model
            if not hasattr(model_value, "encode_image"):
                raise PatchEncoderRuntimeError(
                    "OpenCLIP model does not expose encode_image()."
                )
            feature_tensor = model_value.encode_image(batch_tensor)
            if not isinstance(feature_tensor, torch.Tensor):
                raise PatchEncoderRuntimeError(
                    "encode_image() did not return a torch.Tensor."
                )
            if feature_tensor.ndim != 2:
                raise PatchEncoderRuntimeError(
                    "OpenCLIP feature tensor must be rank-2, "
                    f"got shape={tuple(feature_tensor.shape)}."
                )
            return feature_tensor

        # TIMM fallback.
        model_output: torch.Tensor
        model_value = self._model
        if hasattr(model_value, "forward_features"):
            model_output = model_value.forward_features(batch_tensor)
        else:
            model_output = model_value(batch_tensor)

        if not isinstance(model_output, torch.Tensor):
            raise PatchEncoderRuntimeError(
                "Model forward did not return a torch.Tensor."
            )

        if model_output.ndim == 3:
            # ViT token output [N, tokens, D] -> CLS token.
            model_output = model_output[:, 0, :]

        if model_output.ndim != 2:
            raise PatchEncoderRuntimeError(
                "Model feature tensor must be rank-2, "
                f"got shape={tuple(model_output.shape)}."
            )

        return model_output

    def _coerce_feature_width(
        self, feature_tensor: torch.Tensor, target_width: int
    ) -> torch.Tensor:
        if feature_tensor.ndim != 2:
            raise PatchEncoderRuntimeError(
                f"Feature tensor must be rank-2, got rank={feature_tensor.ndim}."
            )

        current_width: int = int(feature_tensor.shape[1])
        if current_width == target_width:
            return feature_tensor

        if current_width > target_width:
            return feature_tensor[:, :target_width]

        pad_width: int = target_width - current_width
        pad_tensor: torch.Tensor = torch.zeros(
            feature_tensor.shape[0],
            pad_width,
            device=feature_tensor.device,
            dtype=feature_tensor.dtype,
        )
        return torch.cat([feature_tensor, pad_tensor], dim=1)


__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_DEVICE",
    "DEFAULT_PRECISION",
    "DEFAULT_PATCH_SIZE",
    "DEFAULT_INPUT_RESIZE",
    "DEFAULT_TARGET_MAGNIFICATION",
    "DEFAULT_FEATURE_DIM",
    "PatchEncoderError",
    "PatchEncoderConfigError",
    "PatchEncoderInputError",
    "PatchEncoderRuntimeError",
    "PatchEncoder",
    "ConchPatchEncoder",
]
