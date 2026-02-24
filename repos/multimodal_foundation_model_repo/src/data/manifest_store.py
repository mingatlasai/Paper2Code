"""Manifest storage and query layer for THREADS reproduction.

This module implements the design-locked ``ManifestStore`` interface used by
preprocess/pretrain/embed/eval pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import json
import logging

import pandas as pd

from src.data.manifest_schema import (
    DEFAULT_TARGET_MAGNIFICATION_INT,
    MANIFEST_REQUIRED_KEYS,
    ManifestRecord,
    ManifestValidationError,
    parse_manifest_records,
)
from src.utils.io import (
    SUPPORTED_JSON_SUFFIXES,
    SUPPORTED_PARQUET_SUFFIXES,
    SUPPORTED_YAML_SUFFIXES,
    read_json,
    read_parquet,
    read_yaml,
    write_json,
    write_parquet,
    write_yaml,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST_FORMAT: str = "parquet"
DEFAULT_MANIFEST_PATH: str = "data/processed/manifests/pretrain_public_merged.parquet"
DEFAULT_ALLOW_EMPTY_LOAD: bool = False

_DEFAULT_OPTIONAL_FIELD_VALUES: Dict[str, Any] = {
    "rna_path": "",
    "dna_path": "",
    "task_labels": {},
    "meta": {},
    "magnification": DEFAULT_TARGET_MAGNIFICATION_INT,
}

_CORE_REQUIRED_FIELDS: Tuple[str, ...] = (
    "sample_id",
    "patient_id",
    "cohort",
    "slide_path",
)


class ManifestStoreError(Exception):
    """Base exception for manifest-store operations."""


class ManifestStoreLoadError(ManifestStoreError):
    """Raised when loading a manifest fails."""


class ManifestStoreSaveError(ManifestStoreError):
    """Raised when saving a manifest fails."""


class ManifestStoreValidationError(ManifestStoreError):
    """Raised when manifest content violates required constraints."""


@dataclass(frozen=True)
class _ManifestSource:
    """Resolved source metadata for a manifest file."""

    path: Path
    fmt: str


class ManifestStore:
    """In-memory manifest store with deterministic I/O and filtering.

    Public interface (design-locked):
    - ``__init__(manifest_path: str) -> None``
    - ``load() -> list[ManifestRecord]``
    - ``filter_by_cohort(cohorts: list[str]) -> list[ManifestRecord]``
    - ``filter_by_task(task_name: str) -> list[ManifestRecord]``
    - ``upsert(records: list[ManifestRecord]) -> None``
    - ``save(path: str) -> None``
    """

    def __init__(self, manifest_path: str) -> None:
        """Initialize manifest store.

        Args:
            manifest_path: Default manifest file path used by ``load`` when no
                explicit path is provided.
        """
        normalized_manifest_path: str = str(manifest_path).strip()
        if not normalized_manifest_path:
            normalized_manifest_path = DEFAULT_MANIFEST_PATH

        self._manifest_path: str = normalized_manifest_path
        self._records: List[ManifestRecord] = []
        self._index_by_sample_id: Dict[str, int] = {}
        self._is_loaded: bool = False

    def load(self) -> List[ManifestRecord]:
        """Load manifest records from ``self._manifest_path``.

        Returns:
            A validated and deduplicated record list.

        Raises:
            ManifestStoreLoadError: If file I/O or parsing fails.
            ManifestStoreValidationError: If required schema constraints fail.
        """
        source: _ManifestSource = _resolve_manifest_source(self._manifest_path)
        if not source.path.exists():
            if DEFAULT_ALLOW_EMPTY_LOAD:
                self._records = []
                self._index_by_sample_id = {}
                self._is_loaded = True
                return list(self._records)
            raise ManifestStoreLoadError(f"Manifest file not found: {source.path}")

        payload_records: List[Dict[str, Any]]
        try:
            payload_records = _load_payload_records(source)
        except Exception as exc:  # noqa: BLE001
            raise ManifestStoreLoadError(
                f"Failed reading manifest from {source.path}"
            ) from exc

        if not payload_records:
            if DEFAULT_ALLOW_EMPTY_LOAD:
                self._records = []
                self._index_by_sample_id = {}
                self._is_loaded = True
                return list(self._records)
            raise ManifestStoreValidationError(
                f"Manifest contains no records: {source.path}"
            )

        normalized_payload_records: List[Dict[str, Any]] = [
            _normalize_payload_record(record) for record in payload_records
        ]

        try:
            parsed_records: List[ManifestRecord] = parse_manifest_records(
                normalized_payload_records,
                strict_required_keys=True,
            )
        except ManifestValidationError as exc:
            raise ManifestStoreValidationError(
                f"Manifest schema validation failed for {source.path}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ManifestStoreValidationError(
                f"Manifest parsing failed for {source.path}: {exc}"
            ) from exc

        deduplicated_records: List[ManifestRecord] = _deduplicate_keep_first(parsed_records)

        self._records = deduplicated_records
        self._rebuild_index()
        self._is_loaded = True
        return list(self._records)

    def filter_by_cohort(self, cohorts: List[str]) -> List[ManifestRecord]:
        """Filter loaded records by exact cohort match.

        Args:
            cohorts: Cohort names to retain (case-sensitive exact match).

        Returns:
            Filtered records preserving original order.

        Raises:
            ManifestStoreValidationError: If ``cohorts`` is invalid.
        """
        self._ensure_loaded()

        if not isinstance(cohorts, list):
            raise ManifestStoreValidationError(
                f"cohorts must be list[str], got {type(cohorts).__name__}."
            )

        normalized_cohorts: List[str] = [
            _normalize_string(cohort, field_name="cohorts[]", allow_empty=False)
            for cohort in cohorts
        ]
        allowed_cohorts: set[str] = set(normalized_cohorts)

        return [record for record in self._records if record.cohort in allowed_cohorts]

    def filter_by_task(self, task_name: str) -> List[ManifestRecord]:
        """Filter loaded records by task-label key presence.

        Args:
            task_name: Task key to match inside ``record.task_labels``.

        Returns:
            Filtered records preserving original order.
        """
        self._ensure_loaded()
        normalized_task_name: str = _normalize_string(
            task_name,
            field_name="task_name",
            allow_empty=False,
        )

        return [
            record
            for record in self._records
            if normalized_task_name in record.task_labels
        ]

    def upsert(self, records: List[ManifestRecord]) -> None:
        """Insert or replace records keyed by ``sample_id``.

        Existing records keep order; new records are appended in input order.
        For existing ``sample_id``, incoming record replaces the old one.

        Args:
            records: Records to merge into current in-memory store.

        Raises:
            ManifestStoreValidationError: If any incoming record is invalid.
        """
        if not isinstance(records, list):
            raise ManifestStoreValidationError(
                f"records must be list[ManifestRecord], got {type(records).__name__}."
            )

        if not self._is_loaded:
            # Keep behavior deterministic for upsert-first workflows.
            self._records = []
            self._index_by_sample_id = {}
            self._is_loaded = True

        for index, incoming_record in enumerate(records):
            validated_record: ManifestRecord = _coerce_and_validate_manifest_record(
                incoming_record,
                index=index,
            )
            sample_id: str = validated_record.sample_id

            if sample_id in self._index_by_sample_id:
                existing_index: int = self._index_by_sample_id[sample_id]
                self._records[existing_index] = validated_record
            else:
                self._records.append(validated_record)
                self._index_by_sample_id[sample_id] = len(self._records) - 1

        # Rebuild index once in case replacements changed identity unexpectedly.
        self._rebuild_index()

    def save(self, path: str) -> None:
        """Persist current in-memory records.

        Supported output formats: parquet/json/yaml via file suffix.

        Args:
            path: Destination file path.

        Raises:
            ManifestStoreSaveError: If write fails.
            ManifestStoreValidationError: If store is empty or contains invalid data.
        """
        self._ensure_loaded()
        destination_path_raw: str = _normalize_string(
            path,
            field_name="path",
            allow_empty=False,
        )
        destination: _ManifestSource = _resolve_manifest_source(destination_path_raw)

        if not self._records:
            raise ManifestStoreValidationError("Cannot save empty manifest record set.")

        serialized_records: List[Dict[str, Any]] = [
            _normalize_payload_record(record.to_dict()) for record in self._records
        ]

        # Validate by round-tripping through schema parser before write.
        try:
            _ = parse_manifest_records(serialized_records, strict_required_keys=True)
        except Exception as exc:  # noqa: BLE001
            raise ManifestStoreValidationError(
                f"In-memory records failed schema validation before save: {exc}"
            ) from exc

        try:
            _save_payload_records(destination, serialized_records)
        except Exception as exc:  # noqa: BLE001
            raise ManifestStoreSaveError(
                f"Failed to save manifest to {destination.path}"
            ) from exc

    def _ensure_loaded(self) -> None:
        """Ensure store has loaded records once before query operations."""
        if self._is_loaded:
            return
        _ = self.load()

    def _rebuild_index(self) -> None:
        """Rebuild sample-id index and validate uniqueness."""
        index_map: Dict[str, int] = {}
        for index, record in enumerate(self._records):
            sample_id: str = record.sample_id
            if sample_id in index_map:
                raise ManifestStoreValidationError(
                    f"Duplicate sample_id in store state: {sample_id!r}."
                )
            index_map[sample_id] = index
        self._index_by_sample_id = index_map


def _resolve_manifest_source(path: str) -> _ManifestSource:
    """Resolve file path and infer manifest format by suffix."""
    normalized_path: str = _normalize_string(path, field_name="manifest_path", allow_empty=False)
    resolved_path: Path = Path(normalized_path).expanduser().resolve()
    suffix: str = resolved_path.suffix.lower()

    if suffix in SUPPORTED_PARQUET_SUFFIXES:
        fmt: str = "parquet"
    elif suffix in SUPPORTED_JSON_SUFFIXES:
        fmt = "json"
    elif suffix in SUPPORTED_YAML_SUFFIXES:
        fmt = "yaml"
    else:
        raise ManifestStoreValidationError(
            "Unsupported manifest suffix. Expected one of "
            f"{SUPPORTED_PARQUET_SUFFIXES + SUPPORTED_JSON_SUFFIXES + SUPPORTED_YAML_SUFFIXES}, "
            f"got {suffix!r}."
        )

    return _ManifestSource(path=resolved_path, fmt=fmt)


def _load_payload_records(source: _ManifestSource) -> List[Dict[str, Any]]:
    """Load raw manifest records as list[dict]."""
    if source.fmt == "parquet":
        frame: pd.DataFrame = read_parquet(source.path)
        return _dataframe_to_record_dicts(frame)

    if source.fmt == "json":
        payload: Dict[str, Any] = read_json(source.path)
        return _extract_record_list_from_mapping_payload(payload)

    if source.fmt == "yaml":
        payload_yaml: Dict[str, Any] = read_yaml(source.path)
        return _extract_record_list_from_mapping_payload(payload_yaml)

    raise ManifestStoreLoadError(
        f"Unsupported source format {source.fmt!r} for {source.path}."
    )


def _save_payload_records(source: _ManifestSource, records: List[Dict[str, Any]]) -> None:
    """Save raw records using destination format."""
    if source.fmt == "parquet":
        frame: pd.DataFrame = pd.DataFrame.from_records(records)
        write_parquet(frame, source.path, sort_columns=False)
        return

    if source.fmt == "json":
        write_json({"records": records}, source.path)
        return

    if source.fmt == "yaml":
        write_yaml({"records": records}, source.path)
        return

    raise ManifestStoreSaveError(
        f"Unsupported destination format {source.fmt!r} for {source.path}."
    )


def _dataframe_to_record_dicts(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert dataframe to normalized record dictionaries."""
    if frame.empty:
        return []

    records: List[Dict[str, Any]] = frame.to_dict(orient="records")
    return [_normalize_payload_record(record) for record in records]


def _extract_record_list_from_mapping_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract list-of-record payload from mapping-form JSON/YAML.

    Accepted root structures:
    - {"records": [...]} (preferred)
    - {"manifest": [...]} (compat)
    - mapping that itself looks like a single record
    """
    if "records" in payload:
        records_raw: Any = payload["records"]
    elif "manifest" in payload:
        records_raw = payload["manifest"]
    else:
        if _looks_like_single_manifest_record(payload):
            records_raw = [dict(payload)]
        else:
            raise ManifestStoreValidationError(
                "JSON/YAML manifest must contain a 'records' list, 'manifest' list, "
                "or be a single manifest record mapping."
            )

    if not isinstance(records_raw, list):
        raise ManifestStoreValidationError(
            f"Manifest payload list must be list[dict], got {type(records_raw).__name__}."
        )

    normalized_records: List[Dict[str, Any]] = []
    for index, item in enumerate(records_raw):
        if not isinstance(item, Mapping):
            raise ManifestStoreValidationError(
                f"Record at index {index} must be mapping, got {type(item).__name__}."
            )
        normalized_records.append(_normalize_payload_record(dict(item)))

    return normalized_records


def _normalize_payload_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize one raw payload record into schema-compatible mapping."""
    normalized: Dict[str, Any] = dict(record)

    # Fill optional/defaultable fields from config-grounded defaults.
    for field_name, default_value in _DEFAULT_OPTIONAL_FIELD_VALUES.items():
        if field_name not in normalized or normalized[field_name] is None:
            normalized[field_name] = _clone_default_value(default_value)

    # Explicitly enforce all required keys exist.
    missing_required_keys: List[str] = [
        key for key in MANIFEST_REQUIRED_KEYS if key not in normalized
    ]
    if missing_required_keys:
        # For required fields without configured defaults, fail explicitly.
        raise ManifestStoreValidationError(
            f"Record missing required keys: {missing_required_keys}"
        )

    # Enforce non-empty core fields.
    for field_name in _CORE_REQUIRED_FIELDS:
        value: str = _normalize_string(normalized.get(field_name, ""), field_name=field_name, allow_empty=False)
        normalized[field_name] = value

    # Normalize optional path fields.
    normalized["rna_path"] = _normalize_string(
        normalized.get("rna_path", ""),
        field_name="rna_path",
        allow_empty=True,
    )
    normalized["dna_path"] = _normalize_string(
        normalized.get("dna_path", ""),
        field_name="dna_path",
        allow_empty=True,
    )

    # Normalize magnification.
    normalized["magnification"] = _normalize_magnification(normalized.get("magnification"))

    # Normalize dict-like fields.
    normalized["task_labels"] = _normalize_mapping_like(
        normalized.get("task_labels", {}),
        field_name="task_labels",
    )
    normalized["meta"] = _normalize_mapping_like(
        normalized.get("meta", {}),
        field_name="meta",
    )

    # Keep only canonical schema keys in deterministic order.
    return {key: normalized[key] for key in MANIFEST_REQUIRED_KEYS}


def _normalize_mapping_like(value: Any, field_name: str) -> Dict[str, Any]:
    """Normalize mapping-like object, including JSON-string payloads."""
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return {str(key): _normalize_json_scalar(item) for key, item in value.items()}

    if isinstance(value, str):
        stripped_value: str = value.strip()
        if not stripped_value:
            return {}
        try:
            parsed_value: Any = json.loads(stripped_value)
        except json.JSONDecodeError as exc:
            raise ManifestStoreValidationError(
                f"Field '{field_name}' must be mapping or JSON-object string."
            ) from exc
        if not isinstance(parsed_value, Mapping):
            raise ManifestStoreValidationError(
                f"Field '{field_name}' JSON payload must decode to mapping."
            )
        return {str(key): _normalize_json_scalar(item) for key, item in parsed_value.items()}

    raise ManifestStoreValidationError(
        f"Field '{field_name}' must be mapping/string, got {type(value).__name__}."
    )


def _normalize_json_scalar(value: Any) -> Any:
    """Normalize scalar-like values for JSON-safe storage."""
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _normalize_magnification(value: Any) -> int:
    """Normalize magnification to integer form with config-grounded default."""
    if value is None:
        return DEFAULT_TARGET_MAGNIFICATION_INT

    if isinstance(value, bool):
        raise ManifestStoreValidationError("magnification cannot be bool.")

    if isinstance(value, int):
        magnification_int: int = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ManifestStoreValidationError(
                f"magnification must be integer-like, got {value!r}."
            )
        magnification_int = int(value)
    elif isinstance(value, str):
        normalized: str = value.strip().lower()
        if normalized.endswith("x"):
            normalized = normalized[:-1].strip()
        if not normalized:
            magnification_int = DEFAULT_TARGET_MAGNIFICATION_INT
        else:
            try:
                magnification_int = int(normalized)
            except ValueError as exc:
                raise ManifestStoreValidationError(
                    f"Invalid magnification string: {value!r}."
                ) from exc
    else:
        raise ManifestStoreValidationError(
            f"Invalid magnification type: {type(value).__name__}."
        )

    if magnification_int <= 0:
        raise ManifestStoreValidationError(
            f"magnification must be > 0, got {magnification_int}."
        )
    return magnification_int


def _normalize_string(value: Any, field_name: str, allow_empty: bool) -> str:
    """Normalize a value into a stripped string."""
    normalized: str = "" if value is None else str(value).strip()
    if not allow_empty and not normalized:
        raise ManifestStoreValidationError(f"Field '{field_name}' must be non-empty string.")
    return normalized


def _clone_default_value(value: Any) -> Any:
    """Clone mutable defaults to avoid accidental shared state."""
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return list(value)
    return value


def _deduplicate_keep_first(records: Sequence[ManifestRecord]) -> List[ManifestRecord]:
    """Deduplicate by sample_id keeping first stable occurrence."""
    deduplicated: List[ManifestRecord] = []
    seen_sample_ids: set[str] = set()
    dropped_sample_ids: List[str] = []

    for record in records:
        sample_id: str = record.sample_id
        if sample_id in seen_sample_ids:
            dropped_sample_ids.append(sample_id)
            continue
        seen_sample_ids.add(sample_id)
        deduplicated.append(record)

    if dropped_sample_ids:
        unique_dropped_sample_ids: List[str] = sorted(set(dropped_sample_ids))
        LOGGER.warning(
            "Dropped duplicate manifest sample_ids (keeping first occurrence): %s",
            unique_dropped_sample_ids,
        )

    return deduplicated


def _coerce_and_validate_manifest_record(incoming_record: Any, index: int) -> ManifestRecord:
    """Coerce incoming value to ManifestRecord and validate it."""
    if isinstance(incoming_record, ManifestRecord):
        incoming_record.validate()
        return incoming_record

    if isinstance(incoming_record, Mapping):
        normalized_payload: Dict[str, Any] = _normalize_payload_record(incoming_record)
        return ManifestRecord.from_dict(normalized_payload, strict_required_keys=True)

    raise ManifestStoreValidationError(
        f"upsert record at index {index} must be ManifestRecord or mapping, "
        f"got {type(incoming_record).__name__}."
    )


def _looks_like_single_manifest_record(payload: Mapping[str, Any]) -> bool:
    """Return True if mapping contains all required manifest keys."""
    return all(key in payload for key in MANIFEST_REQUIRED_KEYS)


__all__ = [
    "DEFAULT_MANIFEST_FORMAT",
    "DEFAULT_MANIFEST_PATH",
    "DEFAULT_ALLOW_EMPTY_LOAD",
    "ManifestStoreError",
    "ManifestStoreLoadError",
    "ManifestStoreSaveError",
    "ManifestStoreValidationError",
    "ManifestStore",
]
