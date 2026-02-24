## src/data/datasets.py
"""Dataset implementations for TITAN reproduction.

This module provides concrete dataset classes aligned with the design contract:
- BaseDataset
- Stage1Dataset
- Stage2Dataset
- Stage3Dataset
- MultimodalBatch

It consumes prepared artifacts and enforces paper/config invariants:
- patch size: 512
- magnification: 20x
- feature dimension: 768
- stage-1 crop: 16x16 with 2 global (14x14) + 10 local (6x6) views
- stage-3 crop: 64x64
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:
    import h5py
except Exception:  # pragma: no cover - optional at import-time
    h5py = None  # type: ignore[assignment]
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.build_feature_grid import FeatureGrid, FeatureGridBuilder
from src.data.caption_report_processing import TokenizerFactory


# -----------------------------------------------------------------------------
# Config-locked constants (from provided config.yaml and design contract).
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

_DEFAULT_TEXT_MAX_LENGTH: int = 256
_DEFAULT_STAGE1_MIN_VALID_RATIO: float = 0.5
_DEFAULT_POSTERIZATION_LEVELS: int = 16
_DEFAULT_GRID_CACHE_SIZE: int = 32
_DEFAULT_SEED: int = 42
_DEFAULT_LABEL_IGNORE_INDEX: int = -100


class DatasetError(RuntimeError):
    """Base exception for dataset failures."""


class ArtifactError(DatasetError):
    """Raised for missing or malformed artifact files."""


class DatasetSchemaError(DatasetError):
    """Raised when metadata schema requirements are violated."""


class BaseDataset(Dataset):
    """Base dataset contract for TITAN pipelines."""

    def __len__(self) -> int:
        """Return number of samples."""
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        """Return one sample."""
        raise NotImplementedError


@dataclass(frozen=True)
class MultimodalBatch:
    """Multimodal sample contract used by stage-2/3 training."""

    image_grid: FeatureGrid
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    slide_id: str


@dataclass(frozen=True)
class _PairEntry:
    """Internal normalized pair manifest record."""

    pair_id: str
    stage: str
    slide_id: str
    grid_path: str
    text: str
    token_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    labels: Optional[List[int]] = None


class _GridArtifactStore:
    """Shared lazy loader for FeatureGrid artifacts with small in-process cache."""

    def __init__(self, cache_size: int = _DEFAULT_GRID_CACHE_SIZE) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be > 0.")
        self._cache_size: int = int(cache_size)
        self._cache: Dict[str, FeatureGrid] = {}
        self._order: List[str] = []
        self._lock: threading.RLock = threading.RLock()
        self._builder: FeatureGridBuilder = FeatureGridBuilder(patch_size=_PATCH_SIZE_PX)

    def load(self, path: str, slide_id: str = "") -> FeatureGrid:
        """Load a FeatureGrid from `.pt/.pth` or `.h5/.hdf5` artifact."""
        resolved: Path = _resolve_existing_path(path)
        cache_key: str = str(resolved)

        with self._lock:
            if cache_key in self._cache:
                cached: FeatureGrid = self._cache[cache_key]
                return _clone_feature_grid(cached)

        grid: FeatureGrid = self._load_uncached(resolved=resolved, slide_id=slide_id)

        with self._lock:
            self._cache[cache_key] = grid
            self._order.append(cache_key)
            if len(self._order) > self._cache_size:
                oldest_key: str = self._order.pop(0)
                self._cache.pop(oldest_key, None)

        return _clone_feature_grid(grid)

    def _load_uncached(self, resolved: Path, slide_id: str) -> FeatureGrid:
        suffix: str = resolved.suffix.lower()
        if suffix in {".pt", ".pth"}:
            return self._load_pt(path=resolved, slide_id=slide_id)
        if suffix in {".h5", ".hdf5"}:
            return self._load_h5(path=resolved, slide_id=slide_id)
        raise ArtifactError(
            f"Unsupported grid artifact extension '{suffix}' for file: {resolved}."
        )

    def _load_pt(self, path: Path, slide_id: str) -> FeatureGrid:
        try:
            payload: Any = torch.load(path, map_location="cpu")
        except Exception as exc:
            raise ArtifactError(f"Failed loading torch artifact: {path}") from exc

        if isinstance(payload, FeatureGrid):
            loaded_grid: FeatureGrid = payload
            return _validate_feature_grid(_set_slide_id_if_missing(loaded_grid, slide_id))

        if not isinstance(payload, Mapping):
            raise ArtifactError(
                f"PT grid artifact must be FeatureGrid or mapping, got {type(payload).__name__} at {path}."
            )

        if "grid" in payload and isinstance(payload["grid"], FeatureGrid):
            loaded_grid = payload["grid"]
            return _validate_feature_grid(_set_slide_id_if_missing(loaded_grid, slide_id))

        # Common dict formats.
        features_obj: Any = payload.get("features")
        coords_obj: Any = payload.get("coords_xy", payload.get("coords"))
        valid_obj: Any = payload.get("valid_mask", payload.get("mask"))
        stored_slide_id: str = str(payload.get("slide_id", ""))

        if features_obj is None or coords_obj is None:
            raise ArtifactError(
                f"PT grid artifact missing required keys 'features' and 'coords_xy/coords': {path}"
            )

        features_t: torch.Tensor = _to_tensor_float32(features_obj)
        coords_t: torch.Tensor = _to_tensor_int64(coords_obj)

        if features_t.ndim == 2:
            # Likely sparse [N, D] + coords [N, 2]. Build dense grid.
            features_np: np.ndarray = features_t.numpy()
            coords_np: np.ndarray = coords_t.numpy()
            grid: FeatureGrid = self._builder.build_grid(feats=features_np, coords=coords_np)
            grid = _set_slide_id_if_missing(grid, slide_id or stored_slide_id)
            return _validate_feature_grid(grid)

        if features_t.ndim != 3:
            raise ArtifactError(
                f"Unsupported features tensor shape {tuple(features_t.shape)} in PT grid artifact: {path}"
            )

        if coords_t.ndim != 3:
            raise ArtifactError(
                f"coords tensor must have rank 3 for dense grid, got {tuple(coords_t.shape)} at {path}"
            )

        if valid_obj is None:
            valid_t: torch.Tensor = torch.ones(
                (features_t.shape[0], features_t.shape[1]), dtype=torch.bool
            )
        else:
            valid_t = _to_tensor_bool(valid_obj)

        grid = FeatureGrid(
            features=features_t,
            coords_xy=coords_t,
            valid_mask=valid_t,
            slide_id=slide_id or stored_slide_id,
        )
        return _validate_feature_grid(grid)

    def _load_h5(self, path: Path, slide_id: str) -> FeatureGrid:
        if h5py is None:
            raise ArtifactError(
                "h5py is required to read '.h5' feature artifacts but is not available."
            )
        try:
            with h5py.File(path, "r") as h5f:
                if "features" not in h5f or "coords" not in h5f:
                    raise ArtifactError(
                        f"H5 artifact missing required datasets 'features' and 'coords': {path}"
                    )
                features_np: np.ndarray = np.asarray(h5f["features"], dtype=np.float32)
                coords_np: np.ndarray = np.asarray(h5f["coords"], dtype=np.int64)
                attr_slide_id: str = str(h5f.attrs.get("slide_id", ""))
        except ArtifactError:
            raise
        except Exception as exc:
            raise ArtifactError(f"Failed loading HDF5 artifact: {path}") from exc

        if features_np.ndim != 2 or coords_np.ndim != 2:
            raise ArtifactError(
                f"H5 artifacts must store sparse arrays [N,D] and [N,2], got {features_np.shape} and {coords_np.shape} at {path}"
            )

        grid: FeatureGrid = self._builder.build_grid(feats=features_np, coords=coords_np)
        grid = _set_slide_id_if_missing(grid, slide_id or attr_slide_id)
        return _validate_feature_grid(grid)


class Stage1Dataset(BaseDataset):
    """Stage-1 dataset for TITAN-V iBOT pretraining in feature space."""

    def __init__(self, meta_csv: str, groups_path: str, crop_size: int = 16) -> None:
        if not isinstance(meta_csv, str) or not meta_csv.strip():
            raise ValueError("meta_csv must be a non-empty string.")
        if not isinstance(groups_path, str) or not groups_path.strip():
            raise ValueError("groups_path must be a non-empty string.")
        if isinstance(crop_size, bool) or not isinstance(crop_size, (int, np.integer)):
            raise TypeError("crop_size must be an integer.")

        crop_size_int: int = int(crop_size)
        if crop_size_int <= 0:
            raise ValueError("crop_size must be > 0.")

        self.crop_size: int = crop_size_int
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM

        self._global_view_size: Tuple[int, int] = _STAGE1_GLOBAL_GRID
        self._local_view_size: Tuple[int, int] = _STAGE1_LOCAL_GRID

        self._min_valid_ratio: float = _DEFAULT_STAGE1_MIN_VALID_RATIO
        self._posterization_levels: int = _DEFAULT_POSTERIZATION_LEVELS

        seed_offset: int = 101
        seed_value: int = int(os.environ.get("TITAN_DATASET_SEED", str(_DEFAULT_SEED))) + seed_offset
        self._rng: np.random.Generator = np.random.default_rng(seed_value)

        cache_size: int = int(os.environ.get("TITAN_GRID_CACHE_SIZE", str(_DEFAULT_GRID_CACHE_SIZE)))
        self._artifact_store: _GridArtifactStore = _GridArtifactStore(cache_size=cache_size)

        self._meta_csv_path: Path = _resolve_existing_path(meta_csv)
        self._groups_path: Path = _resolve_existing_path(groups_path)

        self._slide_to_grid_path: Dict[str, str] = self._load_stage1_meta(self._meta_csv_path)
        self._groups_by_slide: Dict[str, List[Dict[str, Any]]] = self._load_groups(self._groups_path)

        self._slide_ids: List[str] = sorted(self._slide_to_grid_path.keys())
        if len(self._slide_ids) == 0:
            raise DatasetSchemaError(f"No valid slide rows found in meta CSV: {self._meta_csv_path}")

    def __len__(self) -> int:
        return len(self._slide_ids)

    def __getitem__(self, idx: int) -> dict:
        index: int = _validate_index(idx=idx, size=len(self))
        slide_id: str = self._slide_ids[index]

        region_grid: FeatureGrid = self.sample_region_crop(slide_id=slide_id)
        views: Dict[str, Any] = self.make_ibot_views(region_grid=region_grid)

        sample: Dict[str, Any] = {
            "slide_id": slide_id,
            "region_grid": region_grid,
            "global_views": views["global_views"],
            "local_views": views["local_views"],
            "global_features": views["global_features"],
            "global_coords_xy": views["global_coords_xy"],
            "global_valid_masks": views["global_valid_masks"],
            "local_features": views["local_features"],
            "local_coords_xy": views["local_coords_xy"],
            "local_valid_masks": views["local_valid_masks"],
        }
        return sample

    def sample_region_crop(self, slide_id: str) -> FeatureGrid:
        """Sample one 16x16 region crop from a slide's grouped tissue regions."""
        if not isinstance(slide_id, str) or not slide_id.strip():
            raise ValueError("slide_id must be a non-empty string.")

        normalized_slide_id: str = slide_id.strip()
        if normalized_slide_id not in self._slide_to_grid_path:
            raise KeyError(f"Unknown slide_id '{normalized_slide_id}' in Stage1Dataset.")

        grid_path: str = self._slide_to_grid_path[normalized_slide_id]
        full_grid: FeatureGrid = self._artifact_store.load(path=grid_path, slide_id=normalized_slide_id)

        crop_h: int = self.crop_size
        crop_w: int = self.crop_size
        grid_h: int = int(full_grid.features.shape[0])
        grid_w: int = int(full_grid.features.shape[1])

        if crop_h > grid_h or crop_w > grid_w:
            raise DatasetSchemaError(
                "Stage1 crop cannot fit slide grid. "
                f"slide_id={normalized_slide_id}, crop={(crop_h, crop_w)}, grid={(grid_h, grid_w)}, grid_path={grid_path}"
            )

        group_candidates: List[Dict[str, Any]] = self._resolve_groups_for_slide(normalized_slide_id)
        if len(group_candidates) == 0:
            group_candidates = [self._full_grid_group(full_grid)]

        group_order: np.ndarray = self._rng.permutation(len(group_candidates))
        for candidate_index in group_order.tolist():
            group: Dict[str, Any] = group_candidates[int(candidate_index)]
            bounds: Tuple[int, int, int, int] = self._group_bounds_in_grid(
                group=group,
                grid=full_grid,
            )
            crop: Optional[FeatureGrid] = self._sample_crop_from_bounds(
                grid=full_grid,
                bounds_yx=bounds,
                crop_hw=(crop_h, crop_w),
                min_valid_ratio=self._min_valid_ratio,
            )
            if crop is not None:
                return crop

        # Fallback: exhaustive global search.
        fallback_crop: Optional[FeatureGrid] = self._sample_crop_from_bounds(
            grid=full_grid,
            bounds_yx=(0, grid_h, 0, grid_w),
            crop_hw=(crop_h, crop_w),
            min_valid_ratio=0.0,
            exhaustive_first=True,
        )
        if fallback_crop is not None:
            return fallback_crop

        raise DatasetSchemaError(
            "No valid Stage1 region crop found. "
            f"slide_id={normalized_slide_id}, grid_shape={(grid_h, grid_w)}, grid_path={grid_path}"
        )

    def make_ibot_views(self, region_grid: FeatureGrid) -> dict:
        """Generate iBOT views from a 16x16 region grid.

        Returns a dictionary containing both FeatureGrid objects and stacked
        tensors for trainer/collate compatibility.
        """
        if not isinstance(region_grid, FeatureGrid):
            raise TypeError(f"region_grid must be FeatureGrid, got {type(region_grid).__name__}.")

        expected_h, expected_w = self.crop_size, self.crop_size
        region_h: int = int(region_grid.features.shape[0])
        region_w: int = int(region_grid.features.shape[1])
        if (region_h, region_w) != (expected_h, expected_w):
            raise DatasetSchemaError(
                f"Stage1 region grid must be {(expected_h, expected_w)}, got {(region_h, region_w)}."
            )

        global_views: List[FeatureGrid] = []
        for _ in range(_STAGE1_GLOBAL_VIEWS):
            y0, x0 = self._sample_window_start(
                grid_hw=(region_h, region_w),
                crop_hw=self._global_view_size,
            )
            raw_view: FeatureGrid = _slice_feature_grid(
                grid=region_grid,
                y0=y0,
                y1=y0 + self._global_view_size[0],
                x0=x0,
                x1=x0 + self._global_view_size[1],
            )
            augmented: FeatureGrid = self._augment_feature_grid(raw_view)
            global_views.append(augmented)

        local_views: List[FeatureGrid] = []
        for _ in range(_STAGE1_LOCAL_VIEWS):
            y0, x0 = self._sample_window_start(
                grid_hw=(region_h, region_w),
                crop_hw=self._local_view_size,
            )
            raw_view = _slice_feature_grid(
                grid=region_grid,
                y0=y0,
                y1=y0 + self._local_view_size[0],
                x0=x0,
                x1=x0 + self._local_view_size[1],
            )
            augmented = self._augment_feature_grid(raw_view)
            local_views.append(augmented)

        global_features: torch.Tensor = torch.stack([view.features for view in global_views], dim=0)
        global_coords_xy: torch.Tensor = torch.stack([view.coords_xy for view in global_views], dim=0)
        global_valid_masks: torch.Tensor = torch.stack([view.valid_mask for view in global_views], dim=0)

        local_features: torch.Tensor = torch.stack([view.features for view in local_views], dim=0)
        local_coords_xy: torch.Tensor = torch.stack([view.coords_xy for view in local_views], dim=0)
        local_valid_masks: torch.Tensor = torch.stack([view.valid_mask for view in local_views], dim=0)

        return {
            "global_views": global_views,
            "local_views": local_views,
            "global_features": global_features,
            "global_coords_xy": global_coords_xy,
            "global_valid_masks": global_valid_masks,
            "local_features": local_features,
            "local_coords_xy": local_coords_xy,
            "local_valid_masks": local_valid_masks,
        }

    def _load_stage1_meta(self, path: Path) -> Dict[str, str]:
        frame: pd.DataFrame = pd.read_csv(path)
        if frame.empty:
            return {}

        # Required semantic fields with tolerant column aliases.
        slide_id_col: Optional[str] = _find_first_column(
            frame=frame,
            candidates=("slide_id", "wsi_id", "sample_id", "id"),
        )
        grid_path_col: Optional[str] = _find_first_column(
            frame=frame,
            candidates=("grid_path", "image_grid_path", "artifact_path", "grid", "path"),
        )
        features_path_col: Optional[str] = _find_first_column(
            frame=frame,
            candidates=("features_path", "features_h5", "h5_path"),
        )

        if slide_id_col is None:
            raise DatasetSchemaError(
                f"Stage1 meta CSV missing slide identifier column in {path}."
            )
        if grid_path_col is None and features_path_col is None:
            raise DatasetSchemaError(
                f"Stage1 meta CSV must contain grid path or features path column in {path}."
            )

        base_dir: Path = path.parent
        slide_to_grid: Dict[str, str] = {}

        for _, row in frame.iterrows():
            raw_slide_id: Any = row.get(slide_id_col)
            slide_id: str = str(raw_slide_id).strip() if raw_slide_id is not None else ""
            if not slide_id:
                continue

            grid_path: Optional[str] = None
            if grid_path_col is not None:
                raw_grid_path: Any = row.get(grid_path_col)
                if raw_grid_path is not None and str(raw_grid_path).strip():
                    grid_path = _resolve_maybe_relative(
                        path=str(raw_grid_path).strip(),
                        base_dir=base_dir,
                    )

            if grid_path is None and features_path_col is not None:
                raw_features_path: Any = row.get(features_path_col)
                if raw_features_path is not None and str(raw_features_path).strip():
                    grid_path = _resolve_maybe_relative(
                        path=str(raw_features_path).strip(),
                        base_dir=base_dir,
                    )

            if grid_path is None:
                continue

            if not Path(grid_path).exists():
                continue

            slide_to_grid[slide_id] = grid_path

        return slide_to_grid

    def _load_groups(self, path: Path) -> Dict[str, List[Dict[str, Any]]]:
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)

        groups_by_slide: Dict[str, List[Dict[str, Any]]] = {}

        # Format A: {"slides": {slide_id: [...groups...]}}
        if isinstance(payload, Mapping) and "slides" in payload and isinstance(payload["slides"], Mapping):
            for slide_id_raw, groups_raw in payload["slides"].items():
                slide_id: str = str(slide_id_raw)
                groups_list: List[Dict[str, Any]] = _normalize_group_list(groups_raw)
                if len(groups_list) > 0:
                    groups_by_slide[slide_id] = groups_list
            return groups_by_slide

        # Format B: TissueGrouper-style payload with top-level "groups" list.
        if isinstance(payload, Mapping) and "groups" in payload and isinstance(payload["groups"], list):
            all_groups: List[Dict[str, Any]] = _normalize_group_list(payload["groups"])
            for group in all_groups:
                slide_ids: List[str] = [str(sid) for sid in group.get("slide_ids", []) if str(sid)]
                if len(slide_ids) == 0:
                    groups_by_slide.setdefault("*", []).append(group)
                else:
                    for sid in slide_ids:
                        groups_by_slide.setdefault(sid, []).append(group)
            return groups_by_slide

        # Format C: {slide_id: [groups...]} dictionary.
        if isinstance(payload, Mapping):
            for key, value in payload.items():
                if key in {"format_version", "created_at_utc", "grouping", "num_groups"}:
                    continue
                if isinstance(value, list):
                    normalized: List[Dict[str, Any]] = _normalize_group_list(value)
                    if len(normalized) > 0:
                        groups_by_slide[str(key)] = normalized
            if len(groups_by_slide) > 0:
                return groups_by_slide

        # Format D: plain list.
        if isinstance(payload, list):
            normalized_all: List[Dict[str, Any]] = _normalize_group_list(payload)
            if len(normalized_all) > 0:
                groups_by_slide["*"] = normalized_all
                return groups_by_slide

        return groups_by_slide

    def _resolve_groups_for_slide(self, slide_id: str) -> List[Dict[str, Any]]:
        if slide_id in self._groups_by_slide:
            return self._groups_by_slide[slide_id]
        if "*" in self._groups_by_slide:
            return self._groups_by_slide["*"]
        return []

    def _group_bounds_in_grid(self, group: Mapping[str, Any], grid: FeatureGrid) -> Tuple[int, int, int, int]:
        grid_h: int = int(grid.features.shape[0])
        grid_w: int = int(grid.features.shape[1])

        if "bbox_xyxy" not in group:
            return (0, grid_h, 0, grid_w)

        bbox_value: Any = group.get("bbox_xyxy")
        if not isinstance(bbox_value, (list, tuple)) or len(bbox_value) != 4:
            return (0, grid_h, 0, grid_w)

        x_min_px: float = float(bbox_value[0])
        y_min_px: float = float(bbox_value[1])
        x_max_px: float = float(bbox_value[2])
        y_max_px: float = float(bbox_value[3])

        origin_x: float = float(grid.coords_xy[0, 0, 0].item())
        origin_y: float = float(grid.coords_xy[0, 0, 1].item())

        x0: int = int(np.floor((x_min_px - origin_x) / float(self.patch_size_px)))
        y0: int = int(np.floor((y_min_px - origin_y) / float(self.patch_size_px)))
        x1: int = int(np.ceil((x_max_px - origin_x) / float(self.patch_size_px))) + 1
        y1: int = int(np.ceil((y_max_px - origin_y) / float(self.patch_size_px))) + 1

        x0 = int(np.clip(x0, 0, grid_w))
        y0 = int(np.clip(y0, 0, grid_h))
        x1 = int(np.clip(x1, 0, grid_w))
        y1 = int(np.clip(y1, 0, grid_h))

        if x1 <= x0 or y1 <= y0:
            return (0, grid_h, 0, grid_w)

        return (y0, y1, x0, x1)

    def _sample_crop_from_bounds(
        self,
        grid: FeatureGrid,
        bounds_yx: Tuple[int, int, int, int],
        crop_hw: Tuple[int, int],
        min_valid_ratio: float,
        exhaustive_first: bool = False,
    ) -> Optional[FeatureGrid]:
        y0_b, y1_b, x0_b, x1_b = bounds_yx
        crop_h, crop_w = crop_hw

        if (y1_b - y0_b) < crop_h or (x1_b - x0_b) < crop_w:
            return None

        max_y_start: int = y1_b - crop_h
        max_x_start: int = x1_b - crop_w

        starts: List[Tuple[int, int]] = []
        if exhaustive_first:
            for y_start in range(y0_b, max_y_start + 1):
                for x_start in range(x0_b, max_x_start + 1):
                    starts.append((y_start, x_start))
        else:
            # Random attempts first, then deterministic scan fallback.
            random_attempts: int = 32
            for _ in range(random_attempts):
                y_start = int(self._rng.integers(y0_b, max_y_start + 1))
                x_start = int(self._rng.integers(x0_b, max_x_start + 1))
                starts.append((y_start, x_start))
            for y_start in range(y0_b, max_y_start + 1):
                for x_start in range(x0_b, max_x_start + 1):
                    starts.append((y_start, x_start))

        seen: set[Tuple[int, int]] = set()
        for y_start, x_start in starts:
            key: Tuple[int, int] = (y_start, x_start)
            if key in seen:
                continue
            seen.add(key)

            candidate: FeatureGrid = _slice_feature_grid(
                grid=grid,
                y0=y_start,
                y1=y_start + crop_h,
                x0=x_start,
                x1=x_start + crop_w,
            )
            valid_ratio: float = float(candidate.valid_mask.float().mean().item())
            if valid_ratio >= min_valid_ratio:
                return candidate

        return None

    def _sample_window_start(self, grid_hw: Tuple[int, int], crop_hw: Tuple[int, int]) -> Tuple[int, int]:
        grid_h, grid_w = grid_hw
        crop_h, crop_w = crop_hw

        if crop_h > grid_h or crop_w > grid_w:
            raise DatasetSchemaError(
                f"Cannot sample view crop {crop_hw} from grid {grid_hw}."
            )

        max_y: int = grid_h - crop_h
        max_x: int = grid_w - crop_w

        y0: int = int(self._rng.integers(0, max_y + 1))
        x0: int = int(self._rng.integers(0, max_x + 1))
        return y0, x0

    def _augment_feature_grid(self, grid: FeatureGrid) -> FeatureGrid:
        features: torch.Tensor = grid.features.clone()
        coords: torch.Tensor = grid.coords_xy.clone()
        valid_mask: torch.Tensor = grid.valid_mask.clone()

        # Horizontal flip with p=0.5.
        if bool(self._rng.integers(0, 2)):
            features = torch.flip(features, dims=[1])
            coords = torch.flip(coords, dims=[1])
            valid_mask = torch.flip(valid_mask, dims=[1])

        # Vertical flip with p=0.5.
        if bool(self._rng.integers(0, 2)):
            features = torch.flip(features, dims=[0])
            coords = torch.flip(coords, dims=[0])
            valid_mask = torch.flip(valid_mask, dims=[0])

        # Feature-space posterization with p=0.5.
        if bool(self._rng.integers(0, 2)):
            features = _posterize_features(features=features, levels=self._posterization_levels)

        return FeatureGrid(
            features=features,
            coords_xy=coords,
            valid_mask=valid_mask,
            slide_id=grid.slide_id,
        )

    @staticmethod
    def _full_grid_group(grid: FeatureGrid) -> Dict[str, Any]:
        origin_x: float = float(grid.coords_xy[0, 0, 0].item())
        origin_y: float = float(grid.coords_xy[0, 0, 1].item())
        width: int = int(grid.features.shape[1])
        height: int = int(grid.features.shape[0])
        x_max: float = origin_x + float(width * _PATCH_SIZE_PX)
        y_max: float = origin_y + float(height * _PATCH_SIZE_PX)
        return {
            "bbox_xyxy": [origin_x, origin_y, x_max, y_max],
            "estimated_patch_count": int(height * width),
        }


class Stage2Dataset(BaseDataset):
    """Stage-2 dataset for ROI-caption multimodal alignment."""

    def __init__(self, pairs_csv: str, tokenizer_name: str) -> None:
        if not isinstance(pairs_csv, str) or not pairs_csv.strip():
            raise ValueError("pairs_csv must be a non-empty string.")
        if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
            raise ValueError("tokenizer_name must be a non-empty string.")

        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.region_grid_hw: Tuple[int, int] = _STAGE1_REGION_GRID

        seed_offset: int = 202
        seed_value: int = int(os.environ.get("TITAN_DATASET_SEED", str(_DEFAULT_SEED))) + seed_offset
        self._rng: np.random.Generator = np.random.default_rng(seed_value)

        cache_size: int = int(os.environ.get("TITAN_GRID_CACHE_SIZE", str(_DEFAULT_GRID_CACHE_SIZE)))
        self._artifact_store: _GridArtifactStore = _GridArtifactStore(cache_size=cache_size)

        self._pairs_path: Path = _resolve_existing_path(pairs_csv)
        self._tokenizer = TokenizerFactory.create(
            tokenizer_name=tokenizer_name,
            max_length=_DEFAULT_TEXT_MAX_LENGTH,
        )

        self._entries: List[_PairEntry] = _load_pair_entries(
            path=self._pairs_path,
            expected_stage="stage2",
        )
        if len(self._entries) == 0:
            raise DatasetSchemaError(f"No stage-2 pairs found in manifest: {self._pairs_path}")

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> MultimodalBatch:
        index: int = _validate_index(idx=idx, size=len(self))
        entry: _PairEntry = self._entries[index]

        image_grid: FeatureGrid = self._artifact_store.load(path=entry.grid_path, slide_id=entry.slide_id)
        image_grid = self._fit_or_crop_region_grid(image_grid)

        input_ids, attention_mask, labels = self._build_text_tensors(entry)

        return MultimodalBatch(
            image_grid=image_grid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            slide_id=entry.slide_id,
        )

    def _fit_or_crop_region_grid(self, grid: FeatureGrid) -> FeatureGrid:
        target_h, target_w = self.region_grid_hw
        current_h: int = int(grid.features.shape[0])
        current_w: int = int(grid.features.shape[1])

        if (current_h, current_w) == (target_h, target_w):
            return grid

        if current_h >= target_h and current_w >= target_w:
            y0: int = int(self._rng.integers(0, current_h - target_h + 1))
            x0: int = int(self._rng.integers(0, current_w - target_w + 1))
            return _slice_feature_grid(grid=grid, y0=y0, y1=y0 + target_h, x0=x0, x1=x0 + target_w)

        return _pad_feature_grid(grid=grid, target_hw=(target_h, target_w), patch_size_px=self.patch_size_px)

    def _build_text_tensors(self, entry: _PairEntry) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if entry.token_ids is not None and entry.attention_mask is not None:
            token_ids_list: List[int] = [int(v) for v in entry.token_ids]
            attn_mask_list: List[int] = [int(v) for v in entry.attention_mask]
        else:
            encoded = self._tokenizer.encode(entry.text)
            token_ids_list = [int(v) for v in encoded.token_ids]
            attn_mask_list = [int(v) for v in encoded.attention_mask]

        if len(token_ids_list) != len(attn_mask_list):
            raise DatasetSchemaError(
                f"Token/id mask length mismatch for pair_id={entry.pair_id}, slide_id={entry.slide_id}."
            )

        labels_list: List[int]
        if entry.labels is not None:
            labels_list = [int(v) for v in entry.labels]
        else:
            labels_list = list(token_ids_list)

        if len(labels_list) != len(token_ids_list):
            raise DatasetSchemaError(
                f"Label length mismatch for pair_id={entry.pair_id}, slide_id={entry.slide_id}."
            )

        input_ids: torch.Tensor = torch.tensor(token_ids_list, dtype=torch.long)
        attention_mask: torch.Tensor = torch.tensor(attn_mask_list, dtype=torch.long)
        labels: torch.Tensor = torch.tensor(labels_list, dtype=torch.long)

        return input_ids, attention_mask, labels


class Stage3Dataset(BaseDataset):
    """Stage-3 dataset for WSI-report multimodal alignment."""

    def __init__(
        self,
        pairs_csv: str,
        tokenizer_name: str,
        crop_hw: tuple[int, int] = (64, 64),
    ) -> None:
        if not isinstance(pairs_csv, str) or not pairs_csv.strip():
            raise ValueError("pairs_csv must be a non-empty string.")
        if not isinstance(tokenizer_name, str) or not tokenizer_name.strip():
            raise ValueError("tokenizer_name must be a non-empty string.")
        if not isinstance(crop_hw, tuple) or len(crop_hw) != 2:
            raise TypeError("crop_hw must be tuple[int, int].")

        crop_h_raw, crop_w_raw = crop_hw
        if isinstance(crop_h_raw, bool) or isinstance(crop_w_raw, bool):
            raise TypeError("crop_hw values must be integers.")
        if not isinstance(crop_h_raw, (int, np.integer)) or not isinstance(crop_w_raw, (int, np.integer)):
            raise TypeError("crop_hw values must be integers.")

        crop_h: int = int(crop_h_raw)
        crop_w: int = int(crop_w_raw)
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError("crop_hw values must be > 0.")

        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.crop_hw: Tuple[int, int] = (crop_h, crop_w)

        seed_offset: int = 303
        seed_value: int = int(os.environ.get("TITAN_DATASET_SEED", str(_DEFAULT_SEED))) + seed_offset
        self._rng: np.random.Generator = np.random.default_rng(seed_value)

        cache_size: int = int(os.environ.get("TITAN_GRID_CACHE_SIZE", str(_DEFAULT_GRID_CACHE_SIZE)))
        self._artifact_store: _GridArtifactStore = _GridArtifactStore(cache_size=cache_size)

        self._pairs_path: Path = _resolve_existing_path(pairs_csv)
        self._tokenizer = TokenizerFactory.create(
            tokenizer_name=tokenizer_name,
            max_length=_DEFAULT_TEXT_MAX_LENGTH,
        )

        self._entries: List[_PairEntry] = _load_pair_entries(
            path=self._pairs_path,
            expected_stage="stage3",
        )
        if len(self._entries) == 0:
            raise DatasetSchemaError(f"No stage-3 pairs found in manifest: {self._pairs_path}")

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, idx: int) -> MultimodalBatch:
        index: int = _validate_index(idx=idx, size=len(self))
        entry: _PairEntry = self._entries[index]

        full_grid: FeatureGrid = self._artifact_store.load(path=entry.grid_path, slide_id=entry.slide_id)
        cropped_grid: FeatureGrid = self._random_crop_or_pad(full_grid)

        input_ids, attention_mask, labels = self._build_text_tensors(entry)

        return MultimodalBatch(
            image_grid=cropped_grid,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            slide_id=entry.slide_id,
        )

    def _random_crop_or_pad(self, grid: FeatureGrid) -> FeatureGrid:
        target_h, target_w = self.crop_hw
        current_h: int = int(grid.features.shape[0])
        current_w: int = int(grid.features.shape[1])

        if current_h >= target_h and current_w >= target_w:
            max_y: int = current_h - target_h
            max_x: int = current_w - target_w
            y0: int = int(self._rng.integers(0, max_y + 1))
            x0: int = int(self._rng.integers(0, max_x + 1))
            return _slice_feature_grid(grid=grid, y0=y0, y1=y0 + target_h, x0=x0, x1=x0 + target_w)

        return _pad_feature_grid(grid=grid, target_hw=(target_h, target_w), patch_size_px=self.patch_size_px)

    def _build_text_tensors(self, entry: _PairEntry) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if entry.token_ids is not None and entry.attention_mask is not None:
            token_ids_list: List[int] = [int(v) for v in entry.token_ids]
            attn_mask_list: List[int] = [int(v) for v in entry.attention_mask]
        else:
            encoded = self._tokenizer.encode(entry.text)
            token_ids_list = [int(v) for v in encoded.token_ids]
            attn_mask_list = [int(v) for v in encoded.attention_mask]

        if len(token_ids_list) != len(attn_mask_list):
            raise DatasetSchemaError(
                f"Token/id mask length mismatch for pair_id={entry.pair_id}, slide_id={entry.slide_id}."
            )

        labels_list: List[int]
        if entry.labels is not None:
            labels_list = [int(v) for v in entry.labels]
        else:
            labels_list = list(token_ids_list)

        if len(labels_list) != len(token_ids_list):
            raise DatasetSchemaError(
                f"Label length mismatch for pair_id={entry.pair_id}, slide_id={entry.slide_id}."
            )

        input_ids: torch.Tensor = torch.tensor(token_ids_list, dtype=torch.long)
        attention_mask: torch.Tensor = torch.tensor(attn_mask_list, dtype=torch.long)
        labels: torch.Tensor = torch.tensor(labels_list, dtype=torch.long)

        return input_ids, attention_mask, labels


# -----------------------------------------------------------------------------
# Private utility helpers
# -----------------------------------------------------------------------------

def _validate_index(idx: int, size: int) -> int:
    if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
        raise TypeError(f"Index must be integer, got {type(idx).__name__}.")
    index: int = int(idx)
    if index < 0:
        index += int(size)
    if index < 0 or index >= int(size):
        raise IndexError(f"Index out of range: idx={idx}, size={size}")
    return index


def _resolve_existing_path(path: str) -> Path:
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Path must be a non-empty string.")
    resolved: Path = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    if resolved.is_dir():
        # Prefer standard artifact filenames inside directory.
        candidate_pt: Path = resolved / "grid.pt"
        candidate_h5: Path = resolved / "features.h5"
        if candidate_pt.exists() and candidate_pt.is_file():
            return candidate_pt
        if candidate_h5.exists() and candidate_h5.is_file():
            return candidate_h5
        raise ArtifactError(
            f"Directory path provided but no grid artifact found (expected grid.pt or features.h5): {resolved}"
        )
    return resolved


def _resolve_maybe_relative(path: str, base_dir: Path) -> str:
    candidate: Path = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return str(candidate)


def _find_first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    columns_lower: Dict[str, str] = {str(col).strip().lower(): str(col) for col in frame.columns}
    for candidate in candidates:
        candidate_norm: str = str(candidate).strip().lower()
        if candidate_norm in columns_lower:
            return columns_lower[candidate_norm]
    return None


def _normalize_group_list(groups_raw: Any) -> List[Dict[str, Any]]:
    if not isinstance(groups_raw, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for item in groups_raw:
        if isinstance(item, Mapping):
            normalized.append({str(k): v for k, v in item.items()})
    normalized.sort(key=lambda g: _group_sort_key(g))
    return normalized


def _group_sort_key(group: Mapping[str, Any]) -> Tuple[float, float, float, int]:
    bbox: Any = group.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x_min: float = float(bbox[0])
        y_min: float = float(bbox[1])
    else:
        x_min = 0.0
        y_min = 0.0
    patch_count: float = float(group.get("estimated_patch_count", 0.0))
    group_id: int = int(group.get("group_id", -1))
    return (y_min, x_min, -patch_count, group_id)


def _to_tensor_float32(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.float32)
    array: np.ndarray = np.asarray(value, dtype=np.float32)
    return torch.from_numpy(array)


def _to_tensor_int64(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=torch.int64)
    array: np.ndarray = np.asarray(value, dtype=np.int64)
    return torch.from_numpy(array)


def _to_tensor_bool(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return value.detach().cpu()
        return value.detach().cpu().to(dtype=torch.bool)
    array: np.ndarray = np.asarray(value)
    if array.dtype != np.bool_:
        array = array.astype(np.bool_)
    return torch.from_numpy(array)


def _clone_feature_grid(grid: FeatureGrid) -> FeatureGrid:
    return FeatureGrid(
        features=grid.features.clone(),
        coords_xy=grid.coords_xy.clone(),
        valid_mask=grid.valid_mask.clone(),
        slide_id=grid.slide_id,
    )


def _validate_feature_grid(grid: FeatureGrid) -> FeatureGrid:
    if int(grid.features.shape[-1]) != _FEATURE_DIM:
        raise ArtifactError(
            f"FeatureGrid feature dim must be {_FEATURE_DIM}, got {int(grid.features.shape[-1])}."
        )
    return grid


def _set_slide_id_if_missing(grid: FeatureGrid, slide_id: str) -> FeatureGrid:
    if grid.slide_id:
        return grid
    return FeatureGrid(
        features=grid.features,
        coords_xy=grid.coords_xy,
        valid_mask=grid.valid_mask,
        slide_id=str(slide_id),
    )


def _slice_feature_grid(grid: FeatureGrid, y0: int, y1: int, x0: int, x1: int) -> FeatureGrid:
    if y0 < 0 or x0 < 0:
        raise ValueError(f"Slice start must be non-negative, got y0={y0}, x0={x0}.")
    if y1 <= y0 or x1 <= x0:
        raise ValueError(f"Invalid slice bounds: y=({y0},{y1}), x=({x0},{x1}).")

    h: int = int(grid.features.shape[0])
    w: int = int(grid.features.shape[1])
    if y1 > h or x1 > w:
        raise ValueError(
            f"Slice bounds out of range for grid shape {(h, w)}: y=({y0},{y1}), x=({x0},{x1})."
        )

    return FeatureGrid(
        features=grid.features[y0:y1, x0:x1, :].clone(),
        coords_xy=grid.coords_xy[y0:y1, x0:x1, :].clone(),
        valid_mask=grid.valid_mask[y0:y1, x0:x1].clone(),
        slide_id=grid.slide_id,
    )


def _pad_feature_grid(grid: FeatureGrid, target_hw: Tuple[int, int], patch_size_px: int) -> FeatureGrid:
    target_h, target_w = target_hw
    current_h: int = int(grid.features.shape[0])
    current_w: int = int(grid.features.shape[1])

    if target_h < current_h or target_w < current_w:
        raise ValueError(
            f"Padding target {target_hw} must be >= current {(current_h, current_w)}."
        )

    if (target_h, target_w) == (current_h, current_w):
        return _clone_feature_grid(grid)

    padded_features: torch.Tensor = torch.zeros(
        (target_h, target_w, _FEATURE_DIM),
        dtype=grid.features.dtype,
    )
    padded_coords: torch.Tensor = torch.zeros(
        (target_h, target_w, 2),
        dtype=grid.coords_xy.dtype,
    )
    padded_valid: torch.Tensor = torch.zeros((target_h, target_w), dtype=torch.bool)

    padded_features[:current_h, :current_w, :] = grid.features
    padded_coords[:current_h, :current_w, :] = grid.coords_xy
    padded_valid[:current_h, :current_w] = grid.valid_mask

    # Fill padded coordinates by extrapolating the existing coordinate lattice.
    origin_x: int = int(grid.coords_xy[0, 0, 0].item())
    origin_y: int = int(grid.coords_xy[0, 0, 1].item())

    x_axis: torch.Tensor = torch.arange(target_w, dtype=torch.int64) * int(patch_size_px) + origin_x
    y_axis: torch.Tensor = torch.arange(target_h, dtype=torch.int64) * int(patch_size_px) + origin_y

    padded_coords[:, :, 0] = x_axis.unsqueeze(0).expand(target_h, target_w)
    padded_coords[:, :, 1] = y_axis.unsqueeze(1).expand(target_h, target_w)

    return FeatureGrid(
        features=padded_features,
        coords_xy=padded_coords,
        valid_mask=padded_valid,
        slide_id=grid.slide_id,
    )


def _posterize_features(features: torch.Tensor, levels: int) -> torch.Tensor:
    if levels <= 1:
        return features

    flat: torch.Tensor = features.reshape(-1)
    min_val: torch.Tensor = torch.min(flat)
    max_val: torch.Tensor = torch.max(flat)
    denom: torch.Tensor = max_val - min_val

    if float(torch.abs(denom).item()) < 1e-12:
        return features

    scaled: torch.Tensor = (features - min_val) / denom
    quantized: torch.Tensor = torch.round(scaled * float(levels - 1)) / float(levels - 1)
    restored: torch.Tensor = quantized * denom + min_val
    return restored.to(dtype=features.dtype)


def _load_pair_entries(path: Path, expected_stage: str) -> List[_PairEntry]:
    suffix: str = path.suffix.lower()

    if suffix in {".jsonl", ".json"}:
        raw_rows: List[Dict[str, Any]] = _read_jsonl_rows(path)
    elif suffix in {".csv", ".tsv"}:
        separator: str = "\t" if suffix == ".tsv" else ","
        frame: pd.DataFrame = pd.read_csv(path, sep=separator)
        raw_rows = [{str(k): v for k, v in row.items()} for row in frame.to_dict(orient="records")]
    else:
        raise DatasetSchemaError(
            f"Unsupported pairs manifest extension '{suffix}' for file {path}."
        )

    base_dir: Path = path.parent
    expected_stage_norm: str = expected_stage.strip().lower()

    entries: List[_PairEntry] = []
    for row_index, row in enumerate(raw_rows):
        stage_value: Optional[str] = _extract_stage(row)
        if stage_value is not None and stage_value != expected_stage_norm:
            continue

        slide_id: str = _extract_slide_id(row)
        grid_path_raw: str = _extract_grid_path(row)
        if not slide_id or not grid_path_raw:
            continue

        grid_path_resolved: str = _resolve_maybe_relative(grid_path_raw, base_dir=base_dir)
        if not Path(grid_path_resolved).exists():
            continue

        text_value: str = _extract_text(row=row, expected_stage=expected_stage_norm)
        if not text_value:
            continue

        pair_id: str = _extract_pair_id(row=row, default=f"{expected_stage_norm}_{row_index:08d}")

        token_ids: Optional[List[int]] = _extract_optional_int_list(row, key="token_ids")
        attention_mask: Optional[List[int]] = _extract_optional_int_list(row, key="attention_mask")
        labels: Optional[List[int]] = _extract_optional_int_list(row, key="labels")

        if (token_ids is None) != (attention_mask is None):
            # Skip malformed row, keep dataset strict and safe.
            continue

        entry: _PairEntry = _PairEntry(
            pair_id=pair_id,
            stage=expected_stage_norm,
            slide_id=slide_id,
            grid_path=grid_path_resolved,
            text=text_value,
            token_ids=token_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        entries.append(entry)

    entries.sort(key=lambda item: (item.slide_id, item.pair_id))
    return entries


def _read_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            content: str = line.strip()
            if not content:
                continue
            try:
                payload: Any = json.loads(content)
            except json.JSONDecodeError as exc:
                raise DatasetSchemaError(f"Invalid JSON on line {line_number} in {path}") from exc
            if not isinstance(payload, Mapping):
                raise DatasetSchemaError(
                    f"JSONL row must be object at {path}:{line_number}, got {type(payload).__name__}"
                )
            rows.append({str(k): v for k, v in payload.items()})
    return rows


def _extract_stage(row: Mapping[str, Any]) -> Optional[str]:
    if "stage" not in row or row["stage"] is None:
        return None
    value: str = str(row["stage"]).strip().lower()
    if not value:
        return None
    aliases: Dict[str, str] = {
        "stage2": "stage2",
        "stage_2": "stage2",
        "stage2_roi_caption_alignment": "stage2",
        "roi_caption": "stage2",
        "stage3": "stage3",
        "stage_3": "stage3",
        "stage3_wsi_report_alignment": "stage3",
        "wsi_report": "stage3",
    }
    return aliases.get(value, value)


def _extract_slide_id(row: Mapping[str, Any]) -> str:
    for key in ("slide_id", "wsi_id", "sample_id", "id", "roi_id"):
        if key in row and row[key] is not None:
            value: str = str(row[key]).strip()
            if value:
                return value
    return ""


def _extract_grid_path(row: Mapping[str, Any]) -> str:
    for key in ("grid_path", "image_grid_path", "grid", "artifact_path", "path"):
        if key in row and row[key] is not None:
            value: str = str(row[key]).strip()
            if value:
                return value
    return ""


def _extract_text(row: Mapping[str, Any], expected_stage: str) -> str:
    if expected_stage == "stage2":
        keys: Tuple[str, ...] = ("caption", "text", "description", "roi_caption")
    elif expected_stage == "stage3":
        keys = ("report", "text", "clinical_report", "wsi_report")
    else:
        keys = ("text",)

    for key in keys:
        if key in row and row[key] is not None:
            value: str = str(row[key]).strip()
            if value:
                return value
    return ""


def _extract_pair_id(row: Mapping[str, Any], default: str) -> str:
    if "pair_id" in row and row["pair_id"] is not None:
        pair_id: str = str(row["pair_id"]).strip()
        if pair_id:
            return pair_id
    return default


def _extract_optional_int_list(row: Mapping[str, Any], key: str) -> Optional[List[int]]:
    if key not in row or row[key] is None:
        return None

    value: Any = row[key]
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.tolist()]

    if isinstance(value, str):
        stripped: str = value.strip()
        if not stripped:
            return None
        try:
            parsed: Any = json.loads(stripped)
        except json.JSONDecodeError:
            # Comma-separated fallback.
            parts: List[str] = [item.strip() for item in stripped.split(",") if item.strip()]
            if len(parts) == 0:
                return None
            return [int(item) for item in parts]

        if isinstance(parsed, list):
            return [int(v) for v in parsed]
        return None

    if isinstance(value, (int, np.integer)):
        return [int(value)]

    return None


__all__ = [
    "DatasetError",
    "ArtifactError",
    "DatasetSchemaError",
    "BaseDataset",
    "MultimodalBatch",
    "Stage1Dataset",
    "Stage2Dataset",
    "Stage3Dataset",
]
