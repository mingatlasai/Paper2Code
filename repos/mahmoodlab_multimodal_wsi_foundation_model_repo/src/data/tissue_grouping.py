## src/data/tissue_grouping.py
"""Tissue contour grouping for TITAN stage-1 preprocessing.

This module groups segmented tissue contours by spatial proximity, filters groups
smaller than the paper-required minimum patch count, and saves deterministic
metadata artifacts for stage-1 ROI sampling.

Public interface (design-locked):
- TissueGrouper.__init__(min_patches: int = 16, method: str = "dbscan")
- TissueGrouper.group_contours(contours: list) -> list
- TissueGrouper.filter_small_groups(groups: list) -> list
- TissueGrouper.save_groups(path: str, groups: list) -> None
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


# -----------------------------------------------------------------------------
# Config-locked constants from config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID_SIZE: Tuple[int, int] = (16, 16)
_STAGE1_REGION_SIZE_PX: int = 8192

_DEFAULT_MIN_PATCHES: int = 16
_DEFAULT_METHOD: str = "dbscan"

_ALLOWED_METHODS: Tuple[str, ...] = ("dbscan", "single")

_FORMAT_VERSION: int = 1
_DEFAULT_ENCODING: str = "utf-8"


class TissueGroupingError(RuntimeError):
    """Base exception for tissue grouping failures."""


class TissueGrouper:
    """Group tissue contours and filter small groups.

    Grouping behavior:
    - ``method='dbscan'``: centroid-based DBSCAN with deterministic auto-eps.
    - ``method='single'``: all contours in one group.

    Filtering behavior:
    - Keep groups with ``estimated_patch_count >= min_patches``.
    """

    def __init__(self, min_patches: int = 16, method: str = "dbscan") -> None:
        """Initialize the tissue grouper.

        Args:
            min_patches: Minimum group patch count to keep. Defaults to 16.
            method: Grouping method. Supported: ``dbscan``, ``single``.

        Raises:
            ValueError: If arguments are invalid.
        """
        if isinstance(min_patches, bool) or not isinstance(min_patches, (int, np.integer)):
            raise ValueError("min_patches must be an integer.")
        min_patches_int: int = int(min_patches)
        if min_patches_int <= 0:
            raise ValueError("min_patches must be > 0.")

        method_normalized: str = str(method).strip().lower()
        if method_normalized not in _ALLOWED_METHODS:
            raise ValueError(
                f"Unsupported method '{method}'. Supported methods: {list(_ALLOWED_METHODS)}"
            )

        self.min_patches: int = min_patches_int
        self.method: str = method_normalized

        # Provenance constants used in saved artifacts.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid_size: Tuple[int, int] = _STAGE1_REGION_GRID_SIZE
        self.stage1_region_size_px: int = _STAGE1_REGION_SIZE_PX

    def group_contours(self, contours: list) -> list:
        """Group contours by spatial proximity.

        Args:
            contours: List of contour inputs. Supported item types:
                - ``np.ndarray`` with shape ``[N, 1, 2]`` or ``[N, 2]``.
                - ``dict`` with one of:
                    - ``{"contour": ndarray, ...}``
                    - ``{"points": ndarray/list, ...}``
                  Optional metadata keys:
                    - ``patch_count`` (preferred explicit patch count)
                    - ``slide_id``

        Returns:
            List of group dictionaries sorted deterministically.
        """
        if not isinstance(contours, list):
            raise TypeError(f"contours must be list, got {type(contours).__name__}.")
        if len(contours) == 0:
            return []

        descriptors: List[Dict[str, Any]] = []
        for index, item in enumerate(contours):
            descriptor: Dict[str, Any] = self._build_contour_descriptor(item=item, index=index)
            descriptors.append(descriptor)

        labels: np.ndarray = self._cluster_labels(descriptors)
        groups: List[Dict[str, Any]] = self._assemble_groups(
            descriptors=descriptors,
            labels=labels,
        )
        return groups

    def filter_small_groups(self, groups: list) -> list:
        """Remove groups with fewer than ``self.min_patches`` estimated patches.

        Args:
            groups: List of group dictionaries (from ``group_contours``).

        Returns:
            Filtered group list with reassigned contiguous ``group_id`` values.
        """
        if not isinstance(groups, list):
            raise TypeError(f"groups must be list, got {type(groups).__name__}.")
        if len(groups) == 0:
            return []

        filtered: List[Dict[str, Any]] = []
        for raw_group in groups:
            if not isinstance(raw_group, Mapping):
                raise TypeError("Each group must be a mapping/dictionary.")

            estimated_patch_count: int = int(raw_group.get("estimated_patch_count", 0))
            if estimated_patch_count >= self.min_patches:
                filtered.append(dict(raw_group))

        filtered.sort(key=self._group_sort_key)

        for new_group_id, group in enumerate(filtered):
            group["group_id"] = int(new_group_id)

        return filtered

    def save_groups(self, path: str, groups: list) -> None:
        """Save grouped metadata as deterministic JSON.

        Args:
            path: Output file path.
            groups: Group list to save.
        """
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string.")
        if not isinstance(groups, list):
            raise TypeError(f"groups must be list, got {type(groups).__name__}.")

        output_path: Path = Path(path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        validated_groups: List[Dict[str, Any]] = []
        for index, group in enumerate(groups):
            if not isinstance(group, Mapping):
                raise TypeError(f"groups[{index}] must be mapping, got {type(group).__name__}.")
            validated_groups.append(self._sanitize_group_dict(dict(group)))

        payload: Dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "grouping": {
                "method": self.method,
                "min_patches": self.min_patches,
                "patch_size_px": self.patch_size_px,
                "magnification": self.magnification,
                "stage1_region_grid_size": list(self.stage1_region_grid_size),
                "stage1_region_size_px": self.stage1_region_size_px,
            },
            "num_groups": len(validated_groups),
            "groups": validated_groups,
        }

        rendered: str = json.dumps(
            self._to_json_compatible(payload),
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=output_path.suffix if output_path.suffix else ".json",
            dir=str(output_path.parent),
        )
        os.close(tmp_fd)
        tmp_path: Path = Path(tmp_name)

        try:
            with tmp_path.open("w", encoding=_DEFAULT_ENCODING) as handle:
                handle.write(rendered)
                handle.write("\n")
            os.replace(tmp_path, output_path)
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise TissueGroupingError(f"Failed saving groups to {output_path}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_contour_descriptor(self, item: Any, index: int) -> Dict[str, Any]:
        points_xy: np.ndarray
        meta_patch_count: int | None = None
        slide_id: str = ""

        if isinstance(item, np.ndarray):
            points_xy = self._normalize_points(item)
        elif isinstance(item, Mapping):
            if "contour" in item:
                points_xy = self._normalize_points(item["contour"])
            elif "points" in item:
                points_xy = self._normalize_points(item["points"])
            else:
                raise ValueError(
                    f"contours[{index}] dict must contain 'contour' or 'points' key."
                )

            if "patch_count" in item and item["patch_count"] is not None:
                patch_value: Any = item["patch_count"]
                if isinstance(patch_value, bool) or not isinstance(
                    patch_value,
                    (int, np.integer, float, np.floating),
                ):
                    raise ValueError(f"contours[{index}]['patch_count'] must be numeric.")
                meta_patch_count = max(1, int(round(float(patch_value))))

            if "slide_id" in item and item["slide_id"] is not None:
                slide_id = str(item["slide_id"])
        else:
            raise TypeError(
                f"Unsupported contour type at index {index}: {type(item).__name__}."
            )

        x_values: np.ndarray = points_xy[:, 0]
        y_values: np.ndarray = points_xy[:, 1]

        x_min: float = float(np.min(x_values))
        y_min: float = float(np.min(y_values))
        x_max: float = float(np.max(x_values))
        y_max: float = float(np.max(y_values))

        width: float = max(0.0, x_max - x_min)
        height: float = max(0.0, y_max - y_min)
        bbox_area: float = max(1.0, width * height)

        polygon_area: float = self._polygon_area(points_xy)
        if polygon_area <= 0.0:
            polygon_area = bbox_area

        centroid_x: float = float(np.mean(x_values))
        centroid_y: float = float(np.mean(y_values))

        if meta_patch_count is not None:
            estimated_patch_count: int = meta_patch_count
            patch_count_source: str = "provided"
        else:
            estimated_patch_count = max(1, int(math.ceil(polygon_area / float(self.patch_size_px**2))))
            patch_count_source = "geometry"

        return {
            "contour_index": int(index),
            "slide_id": slide_id,
            "points_xy": points_xy,
            "centroid_xy": (centroid_x, centroid_y),
            "bbox_xyxy": (x_min, y_min, x_max, y_max),
            "bbox_area": float(bbox_area),
            "polygon_area": float(polygon_area),
            "estimated_patch_count": int(estimated_patch_count),
            "patch_count_source": patch_count_source,
        }

    def _cluster_labels(self, descriptors: Sequence[Mapping[str, Any]]) -> np.ndarray:
        num_contours: int = len(descriptors)
        if num_contours == 0:
            return np.zeros((0,), dtype=np.int64)

        if self.method == "single":
            return np.zeros((num_contours,), dtype=np.int64)

        centroids: np.ndarray = np.array(
            [
                [float(desc["centroid_xy"][0]), float(desc["centroid_xy"][1])]
                for desc in descriptors
            ],
            dtype=np.float64,
        )

        if num_contours == 1:
            return np.zeros((1,), dtype=np.int64)

        eps_value: float = self._auto_eps(centroids)
        dbscan: DBSCAN = DBSCAN(eps=eps_value, min_samples=1, metric="euclidean")
        raw_labels: np.ndarray = dbscan.fit_predict(centroids).astype(np.int64, copy=False)

        # Deterministically remap labels to 0..G-1 by spatial ordering.
        unique_labels: List[int] = [int(x) for x in np.unique(raw_labels).tolist()]
        label_to_anchor: Dict[int, Tuple[float, float]] = {}
        for label in unique_labels:
            mask: np.ndarray = raw_labels == np.int64(label)
            subset: np.ndarray = centroids[mask]
            anchor_y: float = float(np.min(subset[:, 1]))
            anchor_x: float = float(np.min(subset[:, 0]))
            label_to_anchor[int(label)] = (anchor_y, anchor_x)

        sorted_labels: List[int] = sorted(unique_labels, key=lambda lab: label_to_anchor[lab])
        remap: Dict[int, int] = {old: new for new, old in enumerate(sorted_labels)}

        remapped: np.ndarray = np.array([remap[int(lab)] for lab in raw_labels], dtype=np.int64)
        return remapped

    def _assemble_groups(
        self,
        descriptors: Sequence[Mapping[str, Any]],
        labels: np.ndarray,
    ) -> List[Dict[str, Any]]:
        if len(descriptors) != int(labels.shape[0]):
            raise TissueGroupingError(
                "Descriptor/label length mismatch: "
                f"len(descriptors)={len(descriptors)}, len(labels)={int(labels.shape[0])}."
            )

        groups_map: MutableMapping[int, List[Mapping[str, Any]]] = {}
        for descriptor, label in zip(descriptors, labels):
            label_int: int = int(label)
            groups_map.setdefault(label_int, []).append(descriptor)

        assembled: List[Dict[str, Any]] = []
        for raw_group_id, members in groups_map.items():
            contour_indices: List[int] = [int(member["contour_index"]) for member in members]

            x_mins: List[float] = [float(member["bbox_xyxy"][0]) for member in members]
            y_mins: List[float] = [float(member["bbox_xyxy"][1]) for member in members]
            x_maxs: List[float] = [float(member["bbox_xyxy"][2]) for member in members]
            y_maxs: List[float] = [float(member["bbox_xyxy"][3]) for member in members]

            x_min: float = float(min(x_mins))
            y_min: float = float(min(y_mins))
            x_max: float = float(max(x_maxs))
            y_max: float = float(max(y_maxs))

            width: float = max(0.0, x_max - x_min)
            height: float = max(0.0, y_max - y_min)
            bbox_area: float = max(1.0, width * height)

            centroid_x: float = float(np.mean([float(m["centroid_xy"][0]) for m in members]))
            centroid_y: float = float(np.mean([float(m["centroid_xy"][1]) for m in members]))

            estimated_patch_count: int = int(
                sum(int(member["estimated_patch_count"]) for member in members)
            )
            if estimated_patch_count <= 0:
                estimated_patch_count = 1

            patch_count_sources: List[str] = sorted(
                {
                    str(member.get("patch_count_source", "unknown"))
                    for member in members
                }
            )

            slide_ids: List[str] = sorted(
                {
                    str(member.get("slide_id", ""))
                    for member in members
                    if str(member.get("slide_id", ""))
                }
            )

            group_record: Dict[str, Any] = {
                "group_id": int(raw_group_id),
                "contour_indices": contour_indices,
                "contour_count": int(len(members)),
                "centroid_xy": [centroid_x, centroid_y],
                "bbox_xyxy": [x_min, y_min, x_max, y_max],
                "width": width,
                "height": height,
                "bbox_area": bbox_area,
                "estimated_patch_count": int(estimated_patch_count),
                "patch_count_sources": patch_count_sources,
                "slide_ids": slide_ids,
                "group_method": self.method,
            }
            assembled.append(group_record)

        assembled.sort(key=self._group_sort_key)
        for new_group_id, group in enumerate(assembled):
            group["group_id"] = int(new_group_id)

        return assembled

    @staticmethod
    def _normalize_points(contour: Any) -> np.ndarray:
        contour_np: np.ndarray = np.asarray(contour)
        if contour_np.ndim == 3 and contour_np.shape[1] == 1 and contour_np.shape[2] == 2:
            points_xy: np.ndarray = contour_np[:, 0, :]
        elif contour_np.ndim == 2 and contour_np.shape[1] == 2:
            points_xy = contour_np
        else:
            raise ValueError(
                "Contour must have shape [N,1,2] or [N,2], "
                f"got {tuple(contour_np.shape)}."
            )

        if points_xy.shape[0] < 3:
            raise ValueError("Contour must contain at least 3 points.")

        points_xy = points_xy.astype(np.float64, copy=False)
        if not np.isfinite(points_xy).all():
            raise ValueError("Contour points contain NaN/Inf values.")

        return points_xy

    @staticmethod
    def _polygon_area(points_xy: np.ndarray) -> float:
        x_vals: np.ndarray = points_xy[:, 0]
        y_vals: np.ndarray = points_xy[:, 1]
        # Shoelace formula with closed polygon via roll.
        area: float = 0.5 * abs(
            float(np.dot(x_vals, np.roll(y_vals, -1)) - np.dot(y_vals, np.roll(x_vals, -1)))
        )
        return float(area)

    @staticmethod
    def _auto_eps(centroids: np.ndarray) -> float:
        """Derive deterministic DBSCAN eps from centroid distances.

        Uses nearest-neighbor distances with a robust percentile. This avoids
        hardcoding unavailable supplementary hyperparameters while remaining
        deterministic for a given contour set.
        """
        if centroids.ndim != 2 or centroids.shape[1] != 2:
            raise ValueError(f"centroids must have shape [N, 2], got {centroids.shape}.")

        num_items: int = int(centroids.shape[0])
        if num_items <= 1:
            return 1.0

        diff: np.ndarray = centroids[:, None, :] - centroids[None, :, :]
        dist_sq: np.ndarray = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)

        nn_dist: np.ndarray = np.sqrt(np.min(dist_sq, axis=1))
        # Robust center + floor to avoid degenerate eps.
        eps: float = float(np.percentile(nn_dist, 60.0))
        if not np.isfinite(eps) or eps <= 0.0:
            eps = float(np.median(nn_dist))
        if not np.isfinite(eps) or eps <= 0.0:
            eps = 1.0

        return eps

    @staticmethod
    def _group_sort_key(group: Mapping[str, Any]) -> Tuple[float, float, float, int]:
        bbox: Sequence[Any] = group.get("bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        y_min: float = float(bbox[1]) if len(bbox) >= 2 else 0.0
        x_min: float = float(bbox[0]) if len(bbox) >= 1 else 0.0
        patch_count: float = float(group.get("estimated_patch_count", 0.0))

        contour_indices: Sequence[Any] = group.get("contour_indices", [])
        first_index: int = int(contour_indices[0]) if len(contour_indices) > 0 else -1

        return (y_min, x_min, -patch_count, first_index)

    def _sanitize_group_dict(self, group: Dict[str, Any]) -> Dict[str, Any]:
        required_keys: Tuple[str, ...] = (
            "group_id",
            "contour_indices",
            "contour_count",
            "centroid_xy",
            "bbox_xyxy",
            "estimated_patch_count",
        )
        for required_key in required_keys:
            if required_key not in group:
                raise ValueError(f"Group missing required key: '{required_key}'.")

        sanitized: Dict[str, Any] = dict(group)
        sanitized["group_id"] = int(sanitized["group_id"])
        sanitized["contour_count"] = int(sanitized["contour_count"])
        sanitized["estimated_patch_count"] = int(sanitized["estimated_patch_count"])

        sanitized["contour_indices"] = [int(v) for v in list(sanitized["contour_indices"])]
        sanitized["centroid_xy"] = [float(v) for v in list(sanitized["centroid_xy"])]
        sanitized["bbox_xyxy"] = [float(v) for v in list(sanitized["bbox_xyxy"])]

        if "width" in sanitized:
            sanitized["width"] = float(sanitized["width"])
        if "height" in sanitized:
            sanitized["height"] = float(sanitized["height"])
        if "bbox_area" in sanitized:
            sanitized["bbox_area"] = float(sanitized["bbox_area"])

        if "slide_ids" in sanitized:
            sanitized["slide_ids"] = [str(v) for v in list(sanitized["slide_ids"])]

        if "patch_count_sources" in sanitized:
            sanitized["patch_count_sources"] = [
                str(v) for v in list(sanitized["patch_count_sources"])
            ]

        if "group_method" in sanitized:
            sanitized["group_method"] = str(sanitized["group_method"])

        return sanitized

    @staticmethod
    def _to_json_compatible(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, Mapping):
            return {str(k): TissueGrouper._to_json_compatible(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [TissueGrouper._to_json_compatible(v) for v in value]
        return str(value)


__all__ = [
    "TissueGroupingError",
    "TissueGrouper",
]
