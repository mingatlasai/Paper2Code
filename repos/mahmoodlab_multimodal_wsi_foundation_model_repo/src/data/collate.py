"""DataLoader collate utilities for TITAN reproduction.

This module implements stage-specific collate functions aligned with the
paper/config constraints and dataset interfaces.

Supported use cases:
- Stage 1 (iBOT): collate multi-view feature-space crops.
- Stage 2/3 (CoCa): collate multimodal samples with padded text and grids.
- Eval embedding/retrieval: collate variable-size feature grids safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.data.build_feature_grid import FeatureGrid
from src.data.datasets import MultimodalBatch


# -----------------------------------------------------------------------------
# Config-locked constants from config.yaml and task contract.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE1_GLOBAL_VIEWS: int = 2
_STAGE1_GLOBAL_GRID: Tuple[int, int] = (14, 14)
_STAGE1_LOCAL_VIEWS: int = 10
_STAGE1_LOCAL_GRID: Tuple[int, int] = (6, 6)

_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DEFAULT_PAD_TOKEN_ID: int = 0
_DEFAULT_LABEL_IGNORE_INDEX: int = -100


class CollateError(RuntimeError):
    """Base exception for collate failures."""


class CollateSchemaError(CollateError):
    """Raised when incoming samples violate expected schema."""


@dataclass(frozen=True)
class EvalCollateOutput:
    """Structured output for evaluation collation.

    Attributes:
        image_features: Tensor [B, H, W, D].
        image_coords_xy: Tensor [B, H, W, 2].
        image_valid_mask: Bool tensor [B, H, W].
        slide_ids: List of slide identifiers.
        labels: Optional tensor [B] or [B, ...] when numeric labels are present.
        extras: Additional per-sample fields preserved as lists.
    """

    image_features: torch.Tensor
    image_coords_xy: torch.Tensor
    image_valid_mask: torch.Tensor
    slide_ids: List[str]
    labels: Optional[torch.Tensor]
    extras: Dict[str, Any]


def collate_stage1_ibot(samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate Stage-1 iBOT samples.

    Expected per-sample keys from ``Stage1Dataset``:
    - ``slide_id``
    - ``region_grid`` (FeatureGrid)
    - ``global_features``: [2, 14, 14, 768]
    - ``global_coords_xy``: [2, 14, 14, 2]
    - ``global_valid_masks``: [2, 14, 14]
    - ``local_features``: [10, 6, 6, 768]
    - ``local_coords_xy``: [10, 6, 6, 2]
    - ``local_valid_masks``: [10, 6, 6]

    Returns:
        Dict[str, Any] with batch dimension added at axis 0.
    """
    sample_list: List[Mapping[str, Any]] = _validate_non_empty_sample_list(samples)

    slide_ids: List[str] = []
    region_features: List[torch.Tensor] = []
    region_coords_xy: List[torch.Tensor] = []
    region_valid_masks: List[torch.Tensor] = []

    global_features: List[torch.Tensor] = []
    global_coords_xy: List[torch.Tensor] = []
    global_valid_masks: List[torch.Tensor] = []

    local_features: List[torch.Tensor] = []
    local_coords_xy: List[torch.Tensor] = []
    local_valid_masks: List[torch.Tensor] = []

    for index, sample in enumerate(sample_list):
        _ensure_mapping(sample=sample, index=index)

        slide_id: str = _get_required_str(sample=sample, key="slide_id", index=index)
        slide_ids.append(slide_id)

        region_grid_obj: Any = sample.get("region_grid")
        if not isinstance(region_grid_obj, FeatureGrid):
            raise CollateSchemaError(
                f"Stage1 sample[{index}] missing valid 'region_grid' FeatureGrid."
            )
        _validate_feature_grid_shape(
            grid=region_grid_obj,
            expected_hw=_STAGE1_REGION_GRID,
            sample_index=index,
            field_name="region_grid",
        )

        region_features.append(region_grid_obj.features.to(dtype=torch.float32))
        region_coords_xy.append(region_grid_obj.coords_xy.to(dtype=torch.int64))
        region_valid_masks.append(region_grid_obj.valid_mask.to(dtype=torch.bool))

        g_feats: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="global_features",
            index=index,
            dtype=torch.float32,
        )
        g_coords: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="global_coords_xy",
            index=index,
            dtype=torch.int64,
        )
        g_mask: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="global_valid_masks",
            index=index,
            dtype=torch.bool,
        )

        l_feats: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="local_features",
            index=index,
            dtype=torch.float32,
        )
        l_coords: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="local_coords_xy",
            index=index,
            dtype=torch.int64,
        )
        l_mask: torch.Tensor = _get_required_tensor(
            sample=sample,
            key="local_valid_masks",
            index=index,
            dtype=torch.bool,
        )

        _validate_stage1_view_tensors(
            global_features=g_feats,
            global_coords_xy=g_coords,
            global_valid_masks=g_mask,
            local_features=l_feats,
            local_coords_xy=l_coords,
            local_valid_masks=l_mask,
            sample_index=index,
        )

        global_features.append(g_feats)
        global_coords_xy.append(g_coords)
        global_valid_masks.append(g_mask)

        local_features.append(l_feats)
        local_coords_xy.append(l_coords)
        local_valid_masks.append(l_mask)

    batch: Dict[str, Any] = {
        "slide_ids": slide_ids,
        "region_features": torch.stack(region_features, dim=0),
        "region_coords_xy": torch.stack(region_coords_xy, dim=0),
        "region_valid_mask": torch.stack(region_valid_masks, dim=0),
        "global_features": torch.stack(global_features, dim=0),
        "global_coords_xy": torch.stack(global_coords_xy, dim=0),
        "global_valid_masks": torch.stack(global_valid_masks, dim=0),
        "local_features": torch.stack(local_features, dim=0),
        "local_coords_xy": torch.stack(local_coords_xy, dim=0),
        "local_valid_masks": torch.stack(local_valid_masks, dim=0),
    }
    return batch


def collate_multimodal(
    samples: Sequence[Union[MultimodalBatch, Mapping[str, Any]]],
    pad_token_id: int = _DEFAULT_PAD_TOKEN_ID,
    label_ignore_index: int = _DEFAULT_LABEL_IGNORE_INDEX,
) -> Dict[str, Any]:
    """Collate Stage-2/Stage-3 multimodal samples.

    Args:
        samples: Sequence of ``MultimodalBatch`` or mapping-compatible samples.
        pad_token_id: Padding token id for ``input_ids``.
        label_ignore_index: Padding label for ignored captioning positions.

    Returns:
        Dict[str, Any] with padded/stacked image and text tensors.
    """
    sample_list: List[Union[MultimodalBatch, Mapping[str, Any]]] = _validate_non_empty_sample_list(samples)

    image_grids: List[FeatureGrid] = []
    input_id_list: List[torch.Tensor] = []
    attention_mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    slide_ids: List[str] = []

    for index, sample in enumerate(sample_list):
        mm_sample: MultimodalBatch = _to_multimodal_batch(sample=sample, index=index)

        _validate_feature_grid_dim(grid=mm_sample.image_grid, sample_index=index, field_name="image_grid")

        input_ids: torch.Tensor = _ensure_1d_long_tensor(mm_sample.input_ids, "input_ids", index)
        attention_mask: torch.Tensor = _ensure_1d_long_tensor(mm_sample.attention_mask, "attention_mask", index)
        labels: torch.Tensor = _ensure_1d_long_tensor(mm_sample.labels, "labels", index)

        if input_ids.shape[0] != attention_mask.shape[0] or input_ids.shape[0] != labels.shape[0]:
            raise CollateSchemaError(
                "Text tensor length mismatch in multimodal sample "
                f"[{index}] slide_id='{mm_sample.slide_id}': "
                f"input_ids={tuple(input_ids.shape)}, attention_mask={tuple(attention_mask.shape)}, "
                f"labels={tuple(labels.shape)}."
            )

        image_grids.append(mm_sample.image_grid)
        input_id_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)
        slide_ids.append(str(mm_sample.slide_id))

    image_features, image_coords_xy, image_valid_mask = _collate_feature_grids(image_grids)
    input_ids_padded, attention_mask_padded, labels_padded = _pad_text_triplet(
        input_ids_list=input_id_list,
        attention_mask_list=attention_mask_list,
        labels_list=labels_list,
        pad_token_id=int(pad_token_id),
        label_ignore_index=int(label_ignore_index),
    )

    batch: Dict[str, Any] = {
        "image_features": image_features,
        "image_coords_xy": image_coords_xy,
        "image_valid_mask": image_valid_mask,
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "labels": labels_padded,
        "slide_ids": slide_ids,
    }
    return batch


def collate_eval_embeddings(
    samples: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Collate variable-size evaluation samples.

    Expected per-sample minimum:
    - ``image_grid`` (or aliases: ``grid``, ``feature_grid``, ``region_grid``)
    Optional:
    - ``slide_id``
    - ``label``/``target``/``y``
    - additional scalar/list metadata fields.

    Returns:
        Dict with padded image tensors, slide ids, optional numeric labels,
        and preserved extra fields.
    """
    sample_list: List[Mapping[str, Any]] = _validate_non_empty_sample_list(samples)

    grids: List[FeatureGrid] = []
    slide_ids: List[str] = []
    label_values: List[Any] = []
    has_label: List[bool] = []

    extras_accumulator: Dict[str, List[Any]] = {}

    for index, sample in enumerate(sample_list):
        _ensure_mapping(sample=sample, index=index)

        grid: FeatureGrid = _extract_feature_grid_from_eval_sample(sample=sample, index=index)
        _validate_feature_grid_dim(grid=grid, sample_index=index, field_name="image_grid")
        grids.append(grid)

        slide_id: str = _extract_slide_id_with_fallback(sample=sample, index=index, default=f"sample_{index}")
        slide_ids.append(slide_id)

        label_present, label_value = _extract_optional_label(sample=sample)
        has_label.append(label_present)
        label_values.append(label_value)

        reserved: set[str] = {
            "image_grid",
            "grid",
            "feature_grid",
            "region_grid",
            "slide_id",
            "label",
            "target",
            "y",
        }
        for key, value in sample.items():
            if key in reserved:
                continue
            extras_accumulator.setdefault(str(key), []).append(value)

    image_features, image_coords_xy, image_valid_mask = _collate_feature_grids(grids)

    labels_tensor: Optional[torch.Tensor] = None
    if all(has_label):
        labels_tensor = _coerce_label_list_to_tensor(label_values)

    output: Dict[str, Any] = {
        "image_features": image_features,
        "image_coords_xy": image_coords_xy,
        "image_valid_mask": image_valid_mask,
        "slide_ids": slide_ids,
        "labels": labels_tensor,
        "extras": extras_accumulator,
    }
    return output


def build_collate_fn(
    mode: str,
    stage: Optional[str] = None,
    pad_token_id: int = _DEFAULT_PAD_TOKEN_ID,
    label_ignore_index: int = _DEFAULT_LABEL_IGNORE_INDEX,
) -> Callable[[Sequence[Any]], Any]:
    """Build a collate callable by mode/stage.

    Args:
        mode: One of ``train_stage1``, ``train_stage2``, ``train_stage3``, ``eval``.
        stage: Optional explicit stage alias.
        pad_token_id: Text padding token id for multimodal modes.
        label_ignore_index: Label ignore value for multimodal modes.

    Returns:
        Callable suitable for ``torch.utils.data.DataLoader(collate_fn=...)``.
    """
    mode_value: str = str(mode).strip().lower()
    stage_value: str = str(stage).strip().lower() if stage is not None else ""

    if mode_value == "train_stage1" or stage_value in {"stage1", "stage1_titan_v"}:
        return collate_stage1_ibot

    if mode_value in {"train_stage2", "train_stage3"} or stage_value in {
        "stage2",
        "stage2_roi_caption_alignment",
        "stage3",
        "stage3_wsi_report_alignment",
    }:

        def _fn(samples: Sequence[Union[MultimodalBatch, Mapping[str, Any]]]) -> Dict[str, Any]:
            return collate_multimodal(
                samples=samples,
                pad_token_id=int(pad_token_id),
                label_ignore_index=int(label_ignore_index),
            )

        return _fn

    if mode_value == "eval":
        return collate_eval_embeddings

    raise CollateError(
        f"Unsupported collate mode='{mode}' stage='{stage}'. "
        "Supported modes: train_stage1, train_stage2, train_stage3, eval."
    )


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

def _validate_non_empty_sample_list(samples: Sequence[Any]) -> List[Any]:
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise CollateSchemaError(f"samples must be a sequence, got {type(samples).__name__}.")
    sample_list: List[Any] = list(samples)
    if len(sample_list) == 0:
        raise CollateSchemaError("Cannot collate an empty batch.")
    return sample_list


def _ensure_mapping(sample: Any, index: int) -> None:
    if not isinstance(sample, Mapping):
        raise CollateSchemaError(
            f"Expected mapping sample at index {index}, got {type(sample).__name__}."
        )


def _get_required_str(sample: Mapping[str, Any], key: str, index: int) -> str:
    value: Any = sample.get(key)
    if value is None:
        raise CollateSchemaError(f"Missing required key '{key}' in sample[{index}].")
    value_str: str = str(value).strip()
    if not value_str:
        raise CollateSchemaError(f"Key '{key}' must be non-empty in sample[{index}].")
    return value_str


def _get_required_tensor(
    sample: Mapping[str, Any],
    key: str,
    index: int,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    value: Any = sample.get(key)
    if value is None:
        raise CollateSchemaError(f"Missing required key '{key}' in sample[{index}].")
    if not isinstance(value, torch.Tensor):
        raise CollateSchemaError(
            f"Key '{key}' in sample[{index}] must be torch.Tensor, got {type(value).__name__}."
        )
    tensor: torch.Tensor = value
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _validate_feature_grid_shape(
    grid: FeatureGrid,
    expected_hw: Tuple[int, int],
    sample_index: int,
    field_name: str,
) -> None:
    _validate_feature_grid_dim(grid=grid, sample_index=sample_index, field_name=field_name)
    expected_h, expected_w = expected_hw
    got_h: int = int(grid.features.shape[0])
    got_w: int = int(grid.features.shape[1])
    if (got_h, got_w) != (expected_h, expected_w):
        raise CollateSchemaError(
            f"{field_name} in sample[{sample_index}] has invalid spatial shape "
            f"{(got_h, got_w)}; expected {(expected_h, expected_w)}."
        )


def _validate_feature_grid_dim(grid: FeatureGrid, sample_index: int, field_name: str) -> None:
    if int(grid.features.shape[-1]) != _FEATURE_DIM:
        raise CollateSchemaError(
            f"{field_name} in sample[{sample_index}] feature dim must be {_FEATURE_DIM}, "
            f"got {int(grid.features.shape[-1])}."
        )
    if grid.coords_xy.ndim != 3 or int(grid.coords_xy.shape[-1]) != 2:
        raise CollateSchemaError(
            f"{field_name} coords_xy in sample[{sample_index}] must have shape [H, W, 2], "
            f"got {tuple(grid.coords_xy.shape)}."
        )
    if grid.valid_mask.ndim != 2:
        raise CollateSchemaError(
            f"{field_name} valid_mask in sample[{sample_index}] must have shape [H, W], "
            f"got {tuple(grid.valid_mask.shape)}."
        )


def _validate_stage1_view_tensors(
    global_features: torch.Tensor,
    global_coords_xy: torch.Tensor,
    global_valid_masks: torch.Tensor,
    local_features: torch.Tensor,
    local_coords_xy: torch.Tensor,
    local_valid_masks: torch.Tensor,
    sample_index: int,
) -> None:
    g_shape_expected: Tuple[int, int, int, int] = (
        _STAGE1_GLOBAL_VIEWS,
        _STAGE1_GLOBAL_GRID[0],
        _STAGE1_GLOBAL_GRID[1],
        _FEATURE_DIM,
    )
    if tuple(global_features.shape) != g_shape_expected:
        raise CollateSchemaError(
            f"global_features in sample[{sample_index}] must be {g_shape_expected}, "
            f"got {tuple(global_features.shape)}."
        )

    g_coords_expected: Tuple[int, int, int, int] = (
        _STAGE1_GLOBAL_VIEWS,
        _STAGE1_GLOBAL_GRID[0],
        _STAGE1_GLOBAL_GRID[1],
        2,
    )
    if tuple(global_coords_xy.shape) != g_coords_expected:
        raise CollateSchemaError(
            f"global_coords_xy in sample[{sample_index}] must be {g_coords_expected}, "
            f"got {tuple(global_coords_xy.shape)}."
        )

    g_mask_expected: Tuple[int, int, int] = (
        _STAGE1_GLOBAL_VIEWS,
        _STAGE1_GLOBAL_GRID[0],
        _STAGE1_GLOBAL_GRID[1],
    )
    if tuple(global_valid_masks.shape) != g_mask_expected:
        raise CollateSchemaError(
            f"global_valid_masks in sample[{sample_index}] must be {g_mask_expected}, "
            f"got {tuple(global_valid_masks.shape)}."
        )

    l_shape_expected: Tuple[int, int, int, int] = (
        _STAGE1_LOCAL_VIEWS,
        _STAGE1_LOCAL_GRID[0],
        _STAGE1_LOCAL_GRID[1],
        _FEATURE_DIM,
    )
    if tuple(local_features.shape) != l_shape_expected:
        raise CollateSchemaError(
            f"local_features in sample[{sample_index}] must be {l_shape_expected}, "
            f"got {tuple(local_features.shape)}."
        )

    l_coords_expected: Tuple[int, int, int, int] = (
        _STAGE1_LOCAL_VIEWS,
        _STAGE1_LOCAL_GRID[0],
        _STAGE1_LOCAL_GRID[1],
        2,
    )
    if tuple(local_coords_xy.shape) != l_coords_expected:
        raise CollateSchemaError(
            f"local_coords_xy in sample[{sample_index}] must be {l_coords_expected}, "
            f"got {tuple(local_coords_xy.shape)}."
        )

    l_mask_expected: Tuple[int, int, int] = (
        _STAGE1_LOCAL_VIEWS,
        _STAGE1_LOCAL_GRID[0],
        _STAGE1_LOCAL_GRID[1],
    )
    if tuple(local_valid_masks.shape) != l_mask_expected:
        raise CollateSchemaError(
            f"local_valid_masks in sample[{sample_index}] must be {l_mask_expected}, "
            f"got {tuple(local_valid_masks.shape)}."
        )


def _to_multimodal_batch(
    sample: Union[MultimodalBatch, Mapping[str, Any]],
    index: int,
) -> MultimodalBatch:
    if isinstance(sample, MultimodalBatch):
        return sample

    if not isinstance(sample, Mapping):
        raise CollateSchemaError(
            f"Expected MultimodalBatch or mapping at index {index}, got {type(sample).__name__}."
        )

    image_grid_obj: Any = sample.get("image_grid")
    input_ids_obj: Any = sample.get("input_ids")
    attention_mask_obj: Any = sample.get("attention_mask")
    labels_obj: Any = sample.get("labels")
    slide_id_obj: Any = sample.get("slide_id")

    if not isinstance(image_grid_obj, FeatureGrid):
        raise CollateSchemaError(
            f"Multimodal sample[{index}] has invalid 'image_grid' type: {type(image_grid_obj).__name__}."
        )

    if not isinstance(input_ids_obj, torch.Tensor):
        raise CollateSchemaError(
            f"Multimodal sample[{index}] missing tensor 'input_ids'."
        )
    if not isinstance(attention_mask_obj, torch.Tensor):
        raise CollateSchemaError(
            f"Multimodal sample[{index}] missing tensor 'attention_mask'."
        )
    if not isinstance(labels_obj, torch.Tensor):
        raise CollateSchemaError(
            f"Multimodal sample[{index}] missing tensor 'labels'."
        )

    slide_id_value: str = str(slide_id_obj) if slide_id_obj is not None else f"sample_{index}"

    return MultimodalBatch(
        image_grid=image_grid_obj,
        input_ids=input_ids_obj,
        attention_mask=attention_mask_obj,
        labels=labels_obj,
        slide_id=slide_id_value,
    )


def _ensure_1d_long_tensor(value: torch.Tensor, name: str, index: int) -> torch.Tensor:
    if value.ndim != 1:
        raise CollateSchemaError(
            f"{name} in sample[{index}] must be rank-1 tensor, got shape {tuple(value.shape)}."
        )
    return value.to(dtype=torch.long)


def _collate_feature_grids(grids: Sequence[FeatureGrid]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(grids) == 0:
        raise CollateSchemaError("Cannot collate empty grid sequence.")

    h_values: List[int] = [int(grid.features.shape[0]) for grid in grids]
    w_values: List[int] = [int(grid.features.shape[1]) for grid in grids]

    max_h: int = max(h_values)
    max_w: int = max(w_values)
    batch_size: int = len(grids)

    dtype_features: torch.dtype = grids[0].features.dtype
    dtype_coords: torch.dtype = grids[0].coords_xy.dtype

    image_features: torch.Tensor = torch.zeros(
        (batch_size, max_h, max_w, _FEATURE_DIM),
        dtype=dtype_features,
    )
    image_coords_xy: torch.Tensor = torch.zeros(
        (batch_size, max_h, max_w, 2),
        dtype=dtype_coords,
    )
    image_valid_mask: torch.Tensor = torch.zeros(
        (batch_size, max_h, max_w),
        dtype=torch.bool,
    )

    for batch_index, grid in enumerate(grids):
        h: int = int(grid.features.shape[0])
        w: int = int(grid.features.shape[1])

        image_features[batch_index, :h, :w, :] = grid.features.to(dtype=dtype_features)
        image_coords_xy[batch_index, :h, :w, :] = grid.coords_xy.to(dtype=dtype_coords)
        image_valid_mask[batch_index, :h, :w] = grid.valid_mask.to(dtype=torch.bool)

    return image_features, image_coords_xy, image_valid_mask


def _pad_text_triplet(
    input_ids_list: Sequence[torch.Tensor],
    attention_mask_list: Sequence[torch.Tensor],
    labels_list: Sequence[torch.Tensor],
    pad_token_id: int,
    label_ignore_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(input_ids_list) == 0:
        raise CollateSchemaError("Cannot pad empty text batch.")
    if not (len(input_ids_list) == len(attention_mask_list) == len(labels_list)):
        raise CollateSchemaError("Text lists must have equal lengths.")

    batch_size: int = len(input_ids_list)
    max_len: int = 0
    for idx, input_ids in enumerate(input_ids_list):
        cur_len: int = int(input_ids.shape[0])
        if cur_len <= 0:
            raise CollateSchemaError(f"input_ids in sample[{idx}] is empty.")
        max_len = max(max_len, cur_len)

    input_ids_padded: torch.Tensor = torch.full(
        (batch_size, max_len),
        fill_value=int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask_padded: torch.Tensor = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
    )
    labels_padded: torch.Tensor = torch.full(
        (batch_size, max_len),
        fill_value=int(label_ignore_index),
        dtype=torch.long,
    )

    for idx, (input_ids, attention_mask, labels) in enumerate(
        zip(input_ids_list, attention_mask_list, labels_list)
    ):
        seq_len: int = int(input_ids.shape[0])
        if int(attention_mask.shape[0]) != seq_len or int(labels.shape[0]) != seq_len:
            raise CollateSchemaError(
                "Mismatched text tensor lengths at sample "
                f"[{idx}]: input_ids={tuple(input_ids.shape)}, "
                f"attention_mask={tuple(attention_mask.shape)}, labels={tuple(labels.shape)}."
            )

        input_ids_padded[idx, :seq_len] = input_ids.to(dtype=torch.long)
        attention_mask_padded[idx, :seq_len] = attention_mask.to(dtype=torch.long)
        labels_padded[idx, :seq_len] = labels.to(dtype=torch.long)

    return input_ids_padded, attention_mask_padded, labels_padded


def _extract_feature_grid_from_eval_sample(sample: Mapping[str, Any], index: int) -> FeatureGrid:
    for key in ("image_grid", "grid", "feature_grid", "region_grid"):
        value: Any = sample.get(key)
        if isinstance(value, FeatureGrid):
            return value
    raise CollateSchemaError(
        f"Eval sample[{index}] missing FeatureGrid under one of: image_grid/grid/feature_grid/region_grid."
    )


def _extract_slide_id_with_fallback(
    sample: Mapping[str, Any],
    index: int,
    default: str,
) -> str:
    for key in ("slide_id", "id", "sample_id"):
        if key in sample and sample[key] is not None:
            value: str = str(sample[key]).strip()
            if value:
                return value
    return default


def _extract_optional_label(sample: Mapping[str, Any]) -> Tuple[bool, Any]:
    for key in ("label", "target", "y"):
        if key in sample:
            return True, sample[key]
    return False, None


def _coerce_label_list_to_tensor(values: Sequence[Any]) -> torch.Tensor:
    if len(values) == 0:
        raise CollateSchemaError("Cannot convert empty label list to tensor.")

    # Accept scalar numeric labels, or already-shaped tensors/arrays.
    first: Any = values[0]

    if isinstance(first, torch.Tensor):
        tensor_list: List[torch.Tensor] = []
        for idx, value in enumerate(values):
            if not isinstance(value, torch.Tensor):
                raise CollateSchemaError(
                    f"Mixed label types: sample[{idx}] is {type(value).__name__}, expected torch.Tensor."
                )
            tensor_list.append(value)
        try:
            return torch.stack(tensor_list, dim=0)
        except Exception as exc:
            raise CollateSchemaError("Failed stacking tensor labels.") from exc

    if isinstance(first, np.ndarray):
        try:
            arr: np.ndarray = np.stack([np.asarray(v) for v in values], axis=0)
            return torch.from_numpy(arr)
        except Exception as exc:
            raise CollateSchemaError("Failed stacking ndarray labels.") from exc

    # Scalar fallback.
    numeric_values: List[float] = []
    all_int_like: bool = True
    for idx, value in enumerate(values):
        if isinstance(value, (int, np.integer, bool, np.bool_)):
            numeric_values.append(float(int(value)))
            continue
        if isinstance(value, (float, np.floating)):
            float_value: float = float(value)
            numeric_values.append(float_value)
            if not float_value.is_integer():
                all_int_like = False
            continue
        raise CollateSchemaError(
            f"Unsupported scalar label type at index {idx}: {type(value).__name__}."
        )

    if all_int_like:
        return torch.tensor([int(round(v)) for v in numeric_values], dtype=torch.long)
    return torch.tensor(numeric_values, dtype=torch.float32)


__all__ = [
    "CollateError",
    "CollateSchemaError",
    "EvalCollateOutput",
    "collate_stage1_ibot",
    "collate_multimodal",
    "collate_eval_embeddings",
    "build_collate_fn",
]
