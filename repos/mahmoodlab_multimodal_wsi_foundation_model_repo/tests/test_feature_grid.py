"""Unit tests for feature-grid construction and cropping.

These tests validate the design-locked interface in ``src/data/build_feature_grid.py``:
- ``FeatureGridBuilder.__init__(patch_size: int = 512)``
- ``FeatureGridBuilder.build_grid(feats: np.ndarray, coords: np.ndarray) -> FeatureGrid``
- ``FeatureGridBuilder.build_background_mask(coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray``
- ``FeatureGridBuilder.crop_grid(grid: FeatureGrid, crop_hw: tuple[int, int], random: bool = True)``

Config-locked constants used by this test module:
- patch size: 512
- feature dim: 768
- stage-1 crop: 16x16
- stage-3 crop: 64x64
"""

from __future__ import annotations

from typing import Final, Tuple

import numpy as np
import pytest
import torch

from src.data.build_feature_grid import FeatureGrid, FeatureGridBuilder, FeatureGridValidationError


_PATCH_SIZE: Final[int] = 512
_FEATURE_DIM: Final[int] = 768
_STAGE1_CROP: Final[Tuple[int, int]] = (16, 16)
_STAGE3_CROP: Final[Tuple[int, int]] = (64, 64)


def _make_builder() -> FeatureGridBuilder:
    """Create a config-aligned feature-grid builder."""
    return FeatureGridBuilder(patch_size=_PATCH_SIZE)


def _make_sparse_inputs() -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic sparse inputs with holes and non-zero origin."""
    coords: np.ndarray = np.asarray(
        [
            [1024, 2048],  # local (0, 0)
            [1536, 2048],  # local (1, 0)
            [1024, 2560],  # local (0, 1)
            [2048, 2560],  # local (2, 1) -> hole at local (1, 1)
        ],
        dtype=np.int64,
    )
    feats: np.ndarray = np.zeros((coords.shape[0], _FEATURE_DIM), dtype=np.float32)
    feats[:, 0] = np.asarray([11.0, 22.0, 33.0, 44.0], dtype=np.float32)
    return feats, coords


def _make_dense_grid(height: int, width: int, slide_id: str = "slide-test") -> FeatureGrid:
    """Create a deterministic dense ``FeatureGrid`` for crop tests."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be > 0.")

    features_np: np.ndarray = np.zeros((height, width, _FEATURE_DIM), dtype=np.float32)
    for y_idx in range(height):
        for x_idx in range(width):
            features_np[y_idx, x_idx, 0] = float(y_idx * 1000 + x_idx)

    x_axis: np.ndarray = np.arange(width, dtype=np.int64) * _PATCH_SIZE
    y_axis: np.ndarray = np.arange(height, dtype=np.int64) * _PATCH_SIZE
    coords_xy_np: np.ndarray = np.zeros((height, width, 2), dtype=np.int64)
    coords_xy_np[:, :, 0] = np.broadcast_to(x_axis.reshape(1, width), (height, width))
    coords_xy_np[:, :, 1] = np.broadcast_to(y_axis.reshape(height, 1), (height, width))

    valid_mask_np: np.ndarray = np.ones((height, width), dtype=np.bool_)
    return FeatureGrid(
        features=torch.from_numpy(features_np),
        coords_xy=torch.from_numpy(coords_xy_np),
        valid_mask=torch.from_numpy(valid_mask_np),
        slide_id=slide_id,
    )


def test_build_grid_maps_sparse_coords_to_correct_dense_indices() -> None:
    """Sparse coordinates should map to expected local dense indices."""
    builder: FeatureGridBuilder = _make_builder()
    feats, coords = _make_sparse_inputs()

    grid: FeatureGrid = builder.build_grid(feats=feats, coords=coords)

    assert tuple(grid.features.shape) == (2, 3, _FEATURE_DIM)
    assert tuple(grid.valid_mask.shape) == (2, 3)
    assert int(grid.valid_mask.sum().item()) == 4

    assert float(grid.features[0, 0, 0].item()) == pytest.approx(11.0)
    assert float(grid.features[0, 1, 0].item()) == pytest.approx(22.0)
    assert float(grid.features[1, 0, 0].item()) == pytest.approx(33.0)
    assert float(grid.features[1, 2, 0].item()) == pytest.approx(44.0)


def test_build_grid_preserves_original_pixel_coords_in_coords_xy() -> None:
    """Valid cells should carry original level-0 pixel coordinates."""
    builder: FeatureGridBuilder = _make_builder()
    feats, coords = _make_sparse_inputs()

    grid: FeatureGrid = builder.build_grid(feats=feats, coords=coords)

    expected_coords: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (1024, 2048),
        (0, 1): (1536, 2048),
        (1, 0): (1024, 2560),
        (1, 2): (2048, 2560),
    }
    for (row_idx, col_idx), (x_exp, y_exp) in expected_coords.items():
        x_value: int = int(grid.coords_xy[row_idx, col_idx, 0].item())
        y_value: int = int(grid.coords_xy[row_idx, col_idx, 1].item())
        assert x_value == x_exp
        assert y_value == y_exp


def test_build_grid_valid_mask_marks_only_observed_patches() -> None:
    """Valid mask should be true exactly at occupied sparse coordinates."""
    builder: FeatureGridBuilder = _make_builder()
    feats, coords = _make_sparse_inputs()

    grid: FeatureGrid = builder.build_grid(feats=feats, coords=coords)

    expected_valid: np.ndarray = np.asarray(
        [
            [True, True, False],
            [True, False, True],
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(grid.valid_mask.cpu().numpy(), expected_valid)

    # Ensure hole cells are not marked valid; feature value may be zero-filled.
    assert bool(grid.valid_mask[1, 1].item()) is False
    assert float(grid.features[1, 1, 0].item()) == pytest.approx(0.0)


def test_build_background_mask_matches_grid_occupancy() -> None:
    """Background mask should be the logical inverse of occupancy."""
    builder: FeatureGridBuilder = _make_builder()
    feats, coords = _make_sparse_inputs()
    grid: FeatureGrid = builder.build_grid(feats=feats, coords=coords)

    background_mask: np.ndarray = builder.build_background_mask(coords=coords, shape=(2, 3))
    assert background_mask.dtype == np.bool_
    assert tuple(background_mask.shape) == (2, 3)

    expected_background: np.ndarray = ~grid.valid_mask.cpu().numpy()
    assert np.array_equal(background_mask, expected_background)


def test_default_stage_crop_hw_matches_config_contract() -> None:
    """Default stage crop helpers should return config-locked shapes."""
    assert FeatureGridBuilder.default_stage1_crop_hw() == _STAGE1_CROP
    assert FeatureGridBuilder.default_stage3_crop_hw() == _STAGE3_CROP


def test_crop_grid_stage1_shape_contract_16x16() -> None:
    """Stage-1 crop should return exact 16x16 spatial shape."""
    builder: FeatureGridBuilder = _make_builder()
    grid: FeatureGrid = _make_dense_grid(height=20, width=19)

    cropped: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=_STAGE1_CROP, random=False)

    assert tuple(cropped.features.shape) == (16, 16, _FEATURE_DIM)
    assert tuple(cropped.coords_xy.shape) == (16, 16, 2)
    assert tuple(cropped.valid_mask.shape) == (16, 16)


def test_crop_grid_stage3_shape_contract_64x64() -> None:
    """Stage-3 crop should return exact 64x64 spatial shape."""
    builder: FeatureGridBuilder = _make_builder()
    grid: FeatureGrid = _make_dense_grid(height=80, width=72)

    cropped: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=_STAGE3_CROP, random=False)

    assert tuple(cropped.features.shape) == (64, 64, _FEATURE_DIM)
    assert tuple(cropped.coords_xy.shape) == (64, 64, 2)
    assert tuple(cropped.valid_mask.shape) == (64, 64)


def test_crop_grid_deterministic_mode_is_reproducible() -> None:
    """Repeated deterministic cropping should be bitwise-identical."""
    builder: FeatureGridBuilder = _make_builder()
    grid: FeatureGrid = _make_dense_grid(height=24, width=25, slide_id="deterministic")

    crop_a: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=(16, 16), random=False)
    crop_b: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=(16, 16), random=False)

    assert torch.equal(crop_a.features, crop_b.features)
    assert torch.equal(crop_a.coords_xy, crop_b.coords_xy)
    assert torch.equal(crop_a.valid_mask, crop_b.valid_mask)
    assert crop_a.slide_id == "deterministic"
    assert crop_b.slide_id == "deterministic"


def test_crop_grid_random_mode_reproducible_with_fixed_seed() -> None:
    """Random cropping should be reproducible when NumPy RNG seed is reset."""
    builder: FeatureGridBuilder = _make_builder()
    grid: FeatureGrid = _make_dense_grid(height=32, width=33)
    crop_hw: tuple[int, int] = (16, 16)

    np.random.seed(7)
    crop_seed_1: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=crop_hw, random=True)

    np.random.seed(7)
    crop_seed_1_repeat: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=crop_hw, random=True)

    np.random.seed(11)
    crop_seed_2: FeatureGrid = builder.crop_grid(grid=grid, crop_hw=crop_hw, random=True)

    assert torch.equal(crop_seed_1.features, crop_seed_1_repeat.features)
    assert torch.equal(crop_seed_1.coords_xy, crop_seed_1_repeat.coords_xy)
    assert torch.equal(crop_seed_1.valid_mask, crop_seed_1_repeat.valid_mask)

    # Different seed should generally produce a different crop window on this grid.
    assert not torch.equal(crop_seed_1.coords_xy, crop_seed_2.coords_xy)


def test_crop_grid_rejects_oversized_crop() -> None:
    """Cropping larger than grid shape should fail explicitly."""
    builder: FeatureGridBuilder = _make_builder()
    grid: FeatureGrid = _make_dense_grid(height=10, width=10)

    with pytest.raises(FeatureGridValidationError):
        _ = builder.crop_grid(grid=grid, crop_hw=(16, 16), random=False)


def test_build_grid_rejects_invalid_coordinate_or_feature_contracts() -> None:
    """Builder should reject invalid shapes, values, and duplicate coordinates."""
    builder: FeatureGridBuilder = _make_builder()

    good_feats: np.ndarray = np.zeros((2, _FEATURE_DIM), dtype=np.float32)
    good_coords: np.ndarray = np.asarray([[0, 0], [512, 0]], dtype=np.int64)

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(feats=np.zeros((2, 16), dtype=np.float32), coords=good_coords)

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(feats=good_feats, coords=np.zeros((2, 3), dtype=np.int64))

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(feats=np.zeros((3, _FEATURE_DIM), dtype=np.float32), coords=good_coords)

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(
            feats=good_feats,
            coords=np.asarray([[1, 0], [512, 0]], dtype=np.int64),
        )

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(
            feats=good_feats,
            coords=np.asarray([[0, -512], [512, 0]], dtype=np.int64),
        )

    with pytest.raises(FeatureGridValidationError):
        _ = builder.build_grid(
            feats=good_feats,
            coords=np.asarray([[0, 0], [0, 0]], dtype=np.int64),
        )
