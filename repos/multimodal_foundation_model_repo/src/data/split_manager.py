"""Split management utilities for THREADS reproduction.

This module implements paper/config-aligned split generation with strict leakage
controls and deterministic behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import logging
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.model_selection import (
    GroupKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)

from src.data.manifest_schema import ManifestRecord
from src.utils.io import read_json, write_json


LOGGER = logging.getLogger(__name__)

DEFAULT_SPLIT_DIR: str = "data/processed/splits"
DEFAULT_SEED: int = 42
DEFAULT_CV_FOLDS: int = 5
DEFAULT_MC_SPLITS: int = 50
DEFAULT_TEST_SIZE: float = 0.2
DEFAULT_GROUP_BY: str = "patient_id"
DEFAULT_FEWSHOT_K_VALUES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
OFFICIAL_SINGLE_FOLD_DATASETS: Tuple[str, ...] = ("EBRAINS", "PANDA", "IMP")

SPLIT_REQUIRED_KEYS: Tuple[str, ...] = (
    "task_name",
    "fold_id",
    "train_ids",
    "test_ids",
)


class SplitManagerError(Exception):
    """Base exception for split manager failures."""


class SplitValidationError(SplitManagerError):
    """Raised when a split payload or parameter is invalid."""


class SplitLoadError(SplitManagerError):
    """Raised when official split loading fails."""


class SplitSaveError(SplitManagerError):
    """Raised when split persistence fails."""


@dataclass(frozen=True)
class _RecordRow:
    """Flattened record row for split computations."""

    sample_id: str
    group_id: str
    label: str


class SplitManager:
    """Build and persist patient-safe deterministic split manifests."""

    def __init__(self, split_dir: str, seed: int) -> None:
        """Initialize split manager.

        Args:
            split_dir: Root directory for split artifacts.
            seed: Base deterministic seed.
        """
        normalized_split_dir: str = str(split_dir).strip() if split_dir is not None else ""
        self._split_dir: Path = Path(
            normalized_split_dir if normalized_split_dir else DEFAULT_SPLIT_DIR
        ).expanduser().resolve()
        self._seed: int = _validate_seed(seed)

    def load_official(self, task_name: str) -> Dict[str, List[str]]:
        """Load one official split payload for a task/dataset.

        Args:
            task_name: Task or dataset key.

        Returns:
            Split mapping with keys `task_name`, `fold_id`, `train_ids`, `test_ids`.

        Raises:
            SplitLoadError: If no valid official split can be resolved.
        """
        normalized_task_name: str = _normalize_non_empty_string(task_name, "task_name")
        candidate_paths: List[Path] = self._official_candidate_paths(normalized_task_name)

        for candidate_path in candidate_paths:
            if not candidate_path.exists() or not candidate_path.is_file():
                continue
            try:
                payload: Dict[str, Any] = read_json(candidate_path)
                split_obj: Dict[str, Any] = self._extract_single_split(payload)
                validated_split: Dict[str, Any] = _validate_and_normalize_split_obj(split_obj)
                return {
                    "task_name": str(validated_split["task_name"]),
                    "fold_id": str(validated_split["fold_id"]),
                    "train_ids": list(validated_split["train_ids"]),
                    "test_ids": list(validated_split["test_ids"]),
                }
            except Exception as exc:  # noqa: BLE001
                raise SplitLoadError(
                    f"Failed to parse official split from {candidate_path}: {exc}"
                ) from exc

        raise SplitLoadError(
            "Official split not found. Tried: "
            + ", ".join(str(path) for path in candidate_paths)
        )

    def make_cv(
        self,
        records: List[ManifestRecord],
        n_folds: int,
        stratify_by: str,
        group_by: str,
    ) -> List[Dict[str, Any]]:
        """Build patient-safe CV splits.

        Args:
            records: Task-filtered manifest records.
            n_folds: Number of CV folds.
            stratify_by: Label source key.
            group_by: Group key (typically `patient_id`).

        Returns:
            List of split dictionaries.
        """
        _validate_positive_int(n_folds, "n_folds", minimum=2)
        rows: List[_RecordRow] = self._build_rows(
            records=records,
            stratify_by=stratify_by,
            group_by=group_by,
        )

        sample_ids: np.ndarray = np.asarray([row.sample_id for row in rows], dtype=object)
        group_ids: np.ndarray = np.asarray([row.group_id for row in rows], dtype=object)
        labels: np.ndarray = np.asarray([row.label for row in rows], dtype=object)

        split_indices: List[Tuple[np.ndarray, np.ndarray]] = self._cv_indices(
            labels=labels,
            group_ids=group_ids,
            n_folds=n_folds,
        )

        task_name: str = _infer_task_name(stratify_by)
        split_objects: List[Dict[str, Any]] = []

        for fold_idx, (train_index, test_index) in enumerate(split_indices):
            train_ids: List[str] = sorted(sample_ids[train_index].tolist())
            test_ids: List[str] = sorted(sample_ids[test_index].tolist())

            label_by_id: Dict[str, str] = {row.sample_id: row.label for row in rows}
            group_by_id: Dict[str, str] = {row.sample_id: row.group_id for row in rows}

            split_obj: Dict[str, Any] = {
                "task_name": task_name,
                "fold_id": str(fold_idx),
                "train_ids": train_ids,
                "test_ids": test_ids,
                "split_type": "cv",
                "split_name": f"cv_fold_{fold_idx}",
                "seed": self._seed,
                "group_by": group_by,
                "stratify_by": stratify_by,
                "label_by_id": label_by_id,
                "group_by_id": group_by_id,
            }
            normalized_split: Dict[str, Any] = _validate_and_normalize_split_obj(split_obj)
            _assert_no_id_overlap(normalized_split)
            _assert_no_group_leakage(normalized_split, group_by_id)
            split_objects.append(normalized_split)

        return split_objects

    def make_monte_carlo(
        self,
        records: List[ManifestRecord],
        n_splits: int,
        test_size: float,
        stratify_by: str,
        group_by: str,
    ) -> List[Dict[str, Any]]:
        """Build repeated group-safe train/test splits.

        Args:
            records: Task-filtered manifest records.
            n_splits: Number of repeated splits.
            test_size: Test fraction in (0, 1).
            stratify_by: Label source key.
            group_by: Group key (typically `patient_id`).

        Returns:
            List of split dictionaries.
        """
        _validate_positive_int(n_splits, "n_splits", minimum=1)
        _validate_test_size(test_size)
        rows: List[_RecordRow] = self._build_rows(
            records=records,
            stratify_by=stratify_by,
            group_by=group_by,
        )

        task_name: str = _infer_task_name(stratify_by)
        label_by_id: Dict[str, str] = {row.sample_id: row.label for row in rows}
        group_by_id: Dict[str, str] = {row.sample_id: row.group_id for row in rows}

        group_to_samples: Dict[str, List[str]] = {}
        group_to_label: Dict[str, str] = {}
        for row in rows:
            group_to_samples.setdefault(row.group_id, []).append(row.sample_id)
            if row.group_id not in group_to_label:
                group_to_label[row.group_id] = row.label
            elif group_to_label[row.group_id] != row.label:
                # Heterogeneous groups cannot be strictly stratified at group level.
                group_to_label[row.group_id] = "__MIXED__"

        unique_groups: List[str] = sorted(group_to_samples.keys())
        _validate_positive_int(len(unique_groups), "number_of_groups", minimum=2)

        group_labels: np.ndarray = np.asarray([group_to_label[group] for group in unique_groups], dtype=object)
        group_array: np.ndarray = np.asarray(unique_groups, dtype=object)

        stratify_possible: bool = self._can_stratify_groups(
            labels=group_labels,
            min_count_per_class=2,
        )

        if stratify_possible:
            splitter: Any = StratifiedShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=self._seed,
            )
            index_generator: Iterable[Tuple[np.ndarray, np.ndarray]] = splitter.split(
                group_array,
                group_labels,
            )
        else:
            splitter = ShuffleSplit(
                n_splits=n_splits,
                test_size=test_size,
                random_state=self._seed,
            )
            index_generator = splitter.split(group_array)

        split_objects: List[Dict[str, Any]] = []

        for split_index, (train_group_index, test_group_index) in enumerate(index_generator):
            train_groups: List[str] = sorted(group_array[train_group_index].tolist())
            test_groups: List[str] = sorted(group_array[test_group_index].tolist())

            train_ids: List[str] = sorted(
                sample_id
                for group in train_groups
                for sample_id in group_to_samples[group]
            )
            test_ids: List[str] = sorted(
                sample_id
                for group in test_groups
                for sample_id in group_to_samples[group]
            )

            split_obj: Dict[str, Any] = {
                "task_name": task_name,
                "fold_id": str(split_index),
                "train_ids": train_ids,
                "test_ids": test_ids,
                "split_type": "mc",
                "split_name": f"mc_split_{split_index}",
                "seed": self._seed,
                "group_by": group_by,
                "stratify_by": stratify_by,
                "test_size": float(test_size),
                "label_by_id": label_by_id,
                "group_by_id": group_by_id,
            }

            normalized_split: Dict[str, Any] = _validate_and_normalize_split_obj(split_obj)
            _assert_no_id_overlap(normalized_split)
            _assert_no_group_leakage(normalized_split, group_by_id)
            split_objects.append(normalized_split)

        return split_objects

    def make_fewshot(self, base_split: Dict[str, Any], k_per_class: int) -> Dict[str, Any]:
        """Derive a few-shot train split from a base split.

        Args:
            base_split: Base split mapping with train/test IDs and label mapping.
            k_per_class: Number of examples per class in the few-shot train subset.

        Returns:
            A new split object with few-shot `train_ids` and unchanged `test_ids`.
        """
        _validate_positive_int(k_per_class, "k_per_class", minimum=1)
        normalized_base: Dict[str, Any] = _validate_and_normalize_split_obj(base_split)

        train_ids: List[str] = list(normalized_base["train_ids"])
        test_ids: List[str] = list(normalized_base["test_ids"])

        label_by_id: Dict[str, str] = self._extract_label_by_id(normalized_base)
        train_label_map: Dict[str, str] = {
            sample_id: label_by_id[sample_id] for sample_id in train_ids if sample_id in label_by_id
        }

        if len(train_label_map) != len(train_ids):
            missing_ids: List[str] = sorted(set(train_ids).difference(set(train_label_map.keys())))
            raise SplitValidationError(
                "Few-shot generation requires labels for all train IDs. "
                f"Missing labels for IDs: {missing_ids[:20]}"
            )

        class_to_ids: Dict[str, List[str]] = {}
        for sample_id in sorted(train_ids):
            class_name: str = train_label_map[sample_id]
            class_to_ids.setdefault(class_name, []).append(sample_id)

        infeasible_classes: List[str] = [
            class_name for class_name, ids in class_to_ids.items() if len(ids) < k_per_class
        ]
        if infeasible_classes:
            raise SplitValidationError(
                "Few-shot request infeasible for one or more classes. "
                f"k={k_per_class}, infeasible_classes={sorted(infeasible_classes)}"
            )

        derived_seed: int = _stable_subseed(
            base_seed=self._seed,
            token=f"fewshot::{normalized_base.get('task_name', '')}::"
            f"{normalized_base.get('fold_id', '')}::k={k_per_class}",
        )
        rng: np.random.Generator = np.random.default_rng(derived_seed)

        selected_ids: List[str] = []
        for class_name in sorted(class_to_ids.keys()):
            class_ids: List[str] = list(class_to_ids[class_name])
            selected_indices: np.ndarray = rng.choice(
                len(class_ids),
                size=k_per_class,
                replace=False,
            )
            selected_class_ids: List[str] = sorted(class_ids[index] for index in selected_indices.tolist())
            selected_ids.extend(selected_class_ids)

        selected_ids = sorted(selected_ids)
        selected_label_by_id: Dict[str, str] = {sample_id: train_label_map[sample_id] for sample_id in selected_ids}

        fewshot_split: Dict[str, Any] = {
            "task_name": str(normalized_base["task_name"]),
            "fold_id": str(normalized_base["fold_id"]),
            "train_ids": selected_ids,
            "test_ids": list(test_ids),
            "split_type": "fewshot",
            "split_name": f"fewshot_k{k_per_class}_{normalized_base.get('fold_id', '')}",
            "seed": self._seed,
            "k_per_class": int(k_per_class),
            "base_fold_id": str(normalized_base.get("fold_id", "")),
            "base_split_type": str(normalized_base.get("split_type", "")),
            "group_by": normalized_base.get("group_by", DEFAULT_GROUP_BY),
            "stratify_by": normalized_base.get("stratify_by", ""),
            "label_by_id": selected_label_by_id,
            "group_by_id": {
                sample_id: group_id
                for sample_id, group_id in normalized_base.get("group_by_id", {}).items()
                if sample_id in selected_ids or sample_id in test_ids
            },
        }

        normalized_fewshot: Dict[str, Any] = _validate_and_normalize_split_obj(fewshot_split)
        _assert_no_id_overlap(normalized_fewshot)

        # Test set must remain unchanged relative to base split.
        if normalized_fewshot["test_ids"] != test_ids:
            raise SplitValidationError("Few-shot test_ids must remain identical to base split test_ids.")

        return normalized_fewshot

    def save_split(self, task_name: str, split_name: str, split_obj: Dict[str, Any]) -> None:
        """Persist a split object under split_dir/task_name/split_name.json.

        Args:
            task_name: Task key namespace.
            split_name: Split identifier.
            split_obj: Split payload.
        """
        normalized_task_name: str = _normalize_non_empty_string(task_name, "task_name")
        normalized_split_name: str = _normalize_non_empty_string(split_name, "split_name")
        normalized_split_obj: Dict[str, Any] = _validate_and_normalize_split_obj(split_obj)

        destination_dir: Path = self._split_dir / normalized_task_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        file_name: str = normalized_split_name
        if not file_name.lower().endswith(".json"):
            file_name = f"{file_name}.json"

        destination_path: Path = destination_dir / file_name
        payload: Dict[str, Any] = {
            "task_name": normalized_task_name,
            "split_name": normalized_split_name,
            "split": normalized_split_obj,
        }

        try:
            write_json(payload, destination_path)
        except Exception as exc:  # noqa: BLE001
            raise SplitSaveError(f"Failed to save split to {destination_path}: {exc}") from exc

    def _official_candidate_paths(self, task_name: str) -> List[Path]:
        """Build candidate file paths for official split loading."""
        lower_task_name: str = task_name.lower()
        paths: List[Path] = [
            self._split_dir / task_name / "official.json",
            self._split_dir / task_name / f"{task_name}.json",
            self._split_dir / task_name / f"{task_name}_official.json",
            self._split_dir / f"{task_name}.json",
            self._split_dir / f"{task_name}_official.json",
            self._split_dir / f"official_{task_name}.json",
            self._split_dir / lower_task_name / "official.json",
            self._split_dir / lower_task_name / f"{lower_task_name}.json",
            self._split_dir / lower_task_name / f"{lower_task_name}_official.json",
            self._split_dir / "official" / f"{task_name}.json",
            self._split_dir / "official" / f"{lower_task_name}.json",
        ]

        unique_paths: List[Path] = []
        seen: set[str] = set()
        for path in paths:
            path_key: str = str(path)
            if path_key in seen:
                continue
            seen.add(path_key)
            unique_paths.append(path)
        return unique_paths

    def _extract_single_split(self, payload: Mapping[str, Any]) -> Dict[str, Any]:
        """Extract a single split object from flexible payload structures."""
        if _looks_like_split(payload):
            return dict(payload)

        if "split" in payload and isinstance(payload["split"], Mapping):
            return dict(payload["split"])

        if "splits" in payload and isinstance(payload["splits"], list):
            splits_list: List[Any] = payload["splits"]
            if not splits_list:
                raise SplitValidationError("Official payload contains an empty 'splits' list.")
            first_item: Any = splits_list[0]
            if not isinstance(first_item, Mapping):
                raise SplitValidationError("First item in 'splits' list is not a mapping.")
            return dict(first_item)

        raise SplitValidationError("Could not extract a split object from payload.")

    def _build_rows(
        self,
        records: Sequence[ManifestRecord],
        stratify_by: str,
        group_by: str,
    ) -> List[_RecordRow]:
        """Normalize and validate records into flattened rows."""
        if not isinstance(records, Sequence) or len(records) == 0:
            raise SplitValidationError("records must be a non-empty sequence of ManifestRecord.")

        normalized_group_by: str = _normalize_non_empty_string(group_by, "group_by")
        normalized_stratify_by: str = _normalize_non_empty_string(stratify_by, "stratify_by")

        rows: List[_RecordRow] = []
        for index, record in enumerate(records):
            if not isinstance(record, ManifestRecord):
                raise SplitValidationError(
                    f"records[{index}] must be ManifestRecord, got {type(record).__name__}."
                )

            sample_id: str = _normalize_non_empty_string(record.sample_id, "sample_id")
            group_value: str = self._resolve_group_value(record, normalized_group_by)
            label_value: str = self._resolve_label_value(record, normalized_stratify_by)

            rows.append(
                _RecordRow(
                    sample_id=sample_id,
                    group_id=group_value,
                    label=label_value,
                )
            )

        rows.sort(key=lambda row: (row.sample_id, row.group_id, row.label))
        _ensure_unique_sample_ids([row.sample_id for row in rows])
        return rows

    def _resolve_group_value(self, record: ManifestRecord, group_by: str) -> str:
        """Resolve group key from record attributes/meta/task_labels."""
        if hasattr(record, group_by):
            value: Any = getattr(record, group_by)
            return _normalize_non_empty_string(value, f"record.{group_by}")

        if group_by in record.meta:
            return _normalize_non_empty_string(record.meta[group_by], f"record.meta[{group_by}]")

        if group_by in record.task_labels:
            return _normalize_non_empty_string(
                record.task_labels[group_by],
                f"record.task_labels[{group_by}]",
            )

        raise SplitValidationError(
            f"group_by key '{group_by}' not found in ManifestRecord fields, meta, or task_labels."
        )

    def _resolve_label_value(self, record: ManifestRecord, stratify_by: str) -> str:
        """Resolve stratification label from record."""
        if hasattr(record, stratify_by):
            value: Any = getattr(record, stratify_by)
            return _normalize_non_empty_string(value, f"record.{stratify_by}")

        if stratify_by in record.task_labels:
            return _normalize_non_empty_string(
                record.task_labels[stratify_by],
                f"record.task_labels[{stratify_by}]",
            )

        if stratify_by in record.meta:
            return _normalize_non_empty_string(
                record.meta[stratify_by],
                f"record.meta[{stratify_by}]",
            )

        if len(record.task_labels) == 1:
            only_label: str = next(iter(record.task_labels.values()))
            return _normalize_non_empty_string(only_label, "record.task_labels.single")

        raise SplitValidationError(
            f"stratify_by key '{stratify_by}' not found in ManifestRecord fields/task_labels/meta."
        )

    def _cv_indices(
        self,
        labels: np.ndarray,
        group_ids: np.ndarray,
        n_folds: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate CV index pairs with stratified-group preference."""
        unique_groups: np.ndarray = np.unique(group_ids)
        if unique_groups.shape[0] < n_folds:
            raise SplitValidationError(
                "Insufficient unique groups for requested CV folds: "
                f"groups={unique_groups.shape[0]}, n_folds={n_folds}."
            )

        stratify_possible: bool = self._can_stratify_samplewise(
            labels=labels,
            groups=group_ids,
            n_folds=n_folds,
        )

        if stratify_possible:
            splitter: Any = StratifiedGroupKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self._seed,
            )
            split_iter: Iterable[Tuple[np.ndarray, np.ndarray]] = splitter.split(
                X=np.zeros(labels.shape[0], dtype=np.float32),
                y=labels,
                groups=group_ids,
            )
        else:
            splitter = GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(
                X=np.zeros(labels.shape[0], dtype=np.float32),
                y=labels,
                groups=group_ids,
            )

        indices: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_idx, test_idx in split_iter:
            indices.append((train_idx, test_idx))

        return indices

    def _can_stratify_samplewise(self, labels: np.ndarray, groups: np.ndarray, n_folds: int) -> bool:
        """Return True when stratified group CV is feasible."""
        if labels.shape[0] == 0 or groups.shape[0] == 0:
            return False

        # Group-level homogeneous label requirement.
        group_to_label: Dict[str, str] = {}
        for label_value, group_value in zip(labels.tolist(), groups.tolist()):
            group_id: str = str(group_value)
            label_id: str = str(label_value)
            if group_id not in group_to_label:
                group_to_label[group_id] = label_id
            elif group_to_label[group_id] != label_id:
                return False

        group_labels: List[str] = [group_to_label[group_id] for group_id in sorted(group_to_label.keys())]
        class_counts: Dict[str, int] = {}
        for label_value in group_labels:
            class_counts[label_value] = class_counts.get(label_value, 0) + 1

        if len(class_counts) < 2:
            return False

        return min(class_counts.values()) >= n_folds

    def _can_stratify_groups(self, labels: np.ndarray, min_count_per_class: int) -> bool:
        """Return True if group labels are valid for stratified shuffle split."""
        if labels.shape[0] == 0:
            return False

        label_values: List[str] = [str(item) for item in labels.tolist()]
        if any(label == "__MIXED__" for label in label_values):
            return False

        class_counts: Dict[str, int] = {}
        for label_value in label_values:
            class_counts[label_value] = class_counts.get(label_value, 0) + 1

        if len(class_counts) < 2:
            return False

        return min(class_counts.values()) >= min_count_per_class

    def _extract_label_by_id(self, split_obj: Mapping[str, Any]) -> Dict[str, str]:
        """Extract label mapping from split payload with backward-compatible keys."""
        if "label_by_id" in split_obj and isinstance(split_obj["label_by_id"], Mapping):
            return {
                str(key): _normalize_non_empty_string(value, f"label_by_id[{key}]", allow_empty=False)
                for key, value in split_obj["label_by_id"].items()
            }

        if "train_label_by_id" in split_obj and isinstance(split_obj["train_label_by_id"], Mapping):
            return {
                str(key): _normalize_non_empty_string(value, f"train_label_by_id[{key}]", allow_empty=False)
                for key, value in split_obj["train_label_by_id"].items()
            }

        if "train_labels" in split_obj and isinstance(split_obj["train_labels"], list):
            train_ids: List[str] = list(split_obj.get("train_ids", []))
            train_labels: List[Any] = list(split_obj["train_labels"])
            if len(train_ids) != len(train_labels):
                raise SplitValidationError(
                    "train_labels length must match train_ids length when used for few-shot generation."
                )
            return {
                sample_id: _normalize_non_empty_string(label, f"train_labels[{sample_id}]", allow_empty=False)
                for sample_id, label in zip(train_ids, train_labels)
            }

        raise SplitValidationError(
            "Few-shot generation requires one of: label_by_id, train_label_by_id, or train_labels+train_ids."
        )


def _infer_task_name(stratify_by: str) -> str:
    """Infer task name for split metadata from stratification key."""
    return _normalize_non_empty_string(stratify_by, "stratify_by")


def _validate_seed(seed: Any) -> int:
    """Validate deterministic seed."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitValidationError(f"seed must be int, got {type(seed).__name__}.")
    if seed < 0:
        raise SplitValidationError(f"seed must be non-negative, got {seed}.")
    return seed


def _validate_positive_int(value: Any, name: str, minimum: int) -> int:
    """Validate integer >= minimum."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SplitValidationError(f"{name} must be int, got {type(value).__name__}.")
    if value < minimum:
        raise SplitValidationError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _validate_test_size(test_size: Any) -> float:
    """Validate test size in (0, 1)."""
    if isinstance(test_size, bool):
        raise SplitValidationError("test_size must be float in (0, 1), got bool.")
    if not isinstance(test_size, (float, int)):
        raise SplitValidationError(
            f"test_size must be float in (0, 1), got {type(test_size).__name__}."
        )
    value: float = float(test_size)
    if not (0.0 < value < 1.0):
        raise SplitValidationError(f"test_size must be in (0, 1), got {value}.")
    return value


def _normalize_non_empty_string(
    value: Any,
    field_name: str,
    allow_empty: bool = False,
) -> str:
    """Normalize string fields with optional emptiness allowance."""
    normalized: str = "" if value is None else str(value).strip()
    if not allow_empty and normalized == "":
        raise SplitValidationError(f"{field_name} must be non-empty.")
    return normalized


def _looks_like_split(payload: Mapping[str, Any]) -> bool:
    """Return True if mapping has required split keys."""
    return all(key in payload for key in SPLIT_REQUIRED_KEYS)


def _validate_and_normalize_split_obj(split_obj: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate split object and normalize ID typing/order."""
    if not isinstance(split_obj, Mapping):
        raise SplitValidationError(f"split_obj must be mapping, got {type(split_obj).__name__}.")

    missing_keys: List[str] = [key for key in SPLIT_REQUIRED_KEYS if key not in split_obj]
    if missing_keys:
        raise SplitValidationError(f"split_obj missing required keys: {missing_keys}")

    task_name: str = _normalize_non_empty_string(split_obj["task_name"], "task_name")
    fold_id: str = _normalize_non_empty_string(split_obj["fold_id"], "fold_id")

    train_ids_raw: Any = split_obj["train_ids"]
    test_ids_raw: Any = split_obj["test_ids"]

    if not isinstance(train_ids_raw, list):
        raise SplitValidationError("train_ids must be list[str].")
    if not isinstance(test_ids_raw, list):
        raise SplitValidationError("test_ids must be list[str].")

    train_ids: List[str] = [_normalize_non_empty_string(sample_id, "train_ids[]") for sample_id in train_ids_raw]
    test_ids: List[str] = [_normalize_non_empty_string(sample_id, "test_ids[]") for sample_id in test_ids_raw]

    if len(train_ids) == 0:
        raise SplitValidationError("train_ids must be non-empty.")
    if len(test_ids) == 0:
        raise SplitValidationError("test_ids must be non-empty.")

    _ensure_unique_sample_ids(train_ids)
    _ensure_unique_sample_ids(test_ids)

    overlap: set[str] = set(train_ids).intersection(set(test_ids))
    if overlap:
        raise SplitValidationError(
            f"train_ids and test_ids overlap (sample leakage): {sorted(overlap)[:20]}"
        )

    normalized_split: Dict[str, Any] = dict(split_obj)
    normalized_split["task_name"] = task_name
    normalized_split["fold_id"] = fold_id
    normalized_split["train_ids"] = train_ids
    normalized_split["test_ids"] = test_ids

    if "label_by_id" in normalized_split:
        label_mapping: Any = normalized_split["label_by_id"]
        if not isinstance(label_mapping, Mapping):
            raise SplitValidationError("label_by_id must be a mapping if present.")
        normalized_split["label_by_id"] = {
            _normalize_non_empty_string(sample_id, "label_by_id.key"): _normalize_non_empty_string(
                label,
                "label_by_id.value",
            )
            for sample_id, label in label_mapping.items()
        }

    if "group_by_id" in normalized_split:
        group_mapping: Any = normalized_split["group_by_id"]
        if not isinstance(group_mapping, Mapping):
            raise SplitValidationError("group_by_id must be a mapping if present.")
        normalized_split["group_by_id"] = {
            _normalize_non_empty_string(sample_id, "group_by_id.key"): _normalize_non_empty_string(
                group_id,
                "group_by_id.value",
            )
            for sample_id, group_id in group_mapping.items()
        }

    return normalized_split


def _ensure_unique_sample_ids(sample_ids: Sequence[str]) -> None:
    """Assert sample IDs are unique."""
    if len(sample_ids) != len(set(sample_ids)):
        raise SplitValidationError("Sample ID list contains duplicates.")


def _assert_no_id_overlap(split_obj: Mapping[str, Any]) -> None:
    """Assert train/test sample disjointness."""
    train_ids: set[str] = set(split_obj["train_ids"])
    test_ids: set[str] = set(split_obj["test_ids"])
    overlap: set[str] = train_ids.intersection(test_ids)
    if overlap:
        raise SplitValidationError(
            f"Detected sample leakage across split partitions: {sorted(overlap)[:20]}"
        )


def _assert_no_group_leakage(split_obj: Mapping[str, Any], group_by_id: Mapping[str, str]) -> None:
    """Assert train/test group disjointness if group mapping is available."""
    if not group_by_id:
        return

    train_groups: set[str] = {
        group_by_id[sample_id]
        for sample_id in split_obj["train_ids"]
        if sample_id in group_by_id
    }
    test_groups: set[str] = {
        group_by_id[sample_id]
        for sample_id in split_obj["test_ids"]
        if sample_id in group_by_id
    }

    overlap: set[str] = train_groups.intersection(test_groups)
    if overlap:
        raise SplitValidationError(
            f"Detected group leakage across split partitions: {sorted(overlap)[:20]}"
        )


def _stable_subseed(base_seed: int, token: str) -> int:
    """Create deterministic derived seed from base seed and token."""
    token_hash: str = hashlib.sha256(token.encode("utf-8")).hexdigest()
    token_value: int = int(token_hash[:16], 16)
    return int((base_seed + token_value) % (2**32))


__all__ = [
    "DEFAULT_SPLIT_DIR",
    "DEFAULT_SEED",
    "DEFAULT_CV_FOLDS",
    "DEFAULT_MC_SPLITS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_GROUP_BY",
    "DEFAULT_FEWSHOT_K_VALUES",
    "OFFICIAL_SINGLE_FOLD_DATASETS",
    "SPLIT_REQUIRED_KEYS",
    "SplitManagerError",
    "SplitValidationError",
    "SplitLoadError",
    "SplitSaveError",
    "SplitManager",
]
