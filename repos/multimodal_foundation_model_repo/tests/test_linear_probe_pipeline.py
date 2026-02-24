"""Unit tests for linear-probe evaluation pipeline contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pytest

from src.data.manifest_schema import ManifestRecord
from src.data.split_manager import (
    DEFAULT_CV_FOLDS,
    DEFAULT_FEWSHOT_K_VALUES,
    DEFAULT_GROUP_BY,
    DEFAULT_SEED as SPLIT_DEFAULT_SEED,
    SplitManager,
    SplitValidationError,
)
from src.eval.linear_probe import (
    DEFAULT_C_VALUE,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_MAX_ITER,
    DEFAULT_SOLVER,
    LinearProbeEvaluator,
)


DEFAULT_TASK_BINARY: str = "synthetic_binary_task"
DEFAULT_TASK_SUBTYPING: str = "synthetic_subtyping_task"
DEFAULT_TASK_GRADING: str = "synthetic_grading_task"
DEFAULT_COHORT: str = "SYNTHETIC"
DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_SEED: int = 42
DEFAULT_SLIDES_PER_PATIENT: int = 2

TASK_TYPE_BINARY: str = "binary_classification"
TASK_TYPE_SUBTYPING: str = "subtyping_multiclass"
TASK_TYPE_GRADING: str = "grading_multiclass"

METRIC_MACRO_AUC: str = "macro_auc"
METRIC_BACC: str = "balanced_accuracy"
METRIC_QWK: str = "quadratic_weighted_kappa"


@dataclass(frozen=True)
class _EvalResult:
    """Container for a single fold evaluation output."""

    fold: str
    metric_name: str
    value: float
    task: str
    model_name: str

    def to_row(self) -> Dict[str, Any]:
        """Convert to metrics row shape used by eval reporting."""
        return {
            "task": self.task,
            "fold": self.fold,
            "metric_name": self.metric_name,
            "value": float(self.value),
            "model_name": self.model_name,
        }


def test_constants_match_config_policy() -> None:
    """Validate config-locked defaults used by linear-probe tests."""
    assert DEFAULT_EMBEDDING_DIM == 1024
    assert DEFAULT_C_VALUE == 0.5
    assert DEFAULT_SOLVER == "lbfgs"
    assert DEFAULT_MAX_ITER == 10000
    assert DEFAULT_CLASS_WEIGHT == "balanced"
    assert DEFAULT_CV_FOLDS == 5
    assert DEFAULT_FEWSHOT_K_VALUES == (1, 2, 4, 8, 16, 32)


def test_linear_probe_binary_smoke_valid_metric(tmp_path: Any) -> None:
    """Binary smoke test: valid split/embeddings produce bounded AUC."""
    class_names: Tuple[str, ...] = ("0", "1")
    records: List[ManifestRecord] = _make_records(
        task_name=DEFAULT_TASK_BINARY,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=DEFAULT_TASK_BINARY, split_dir=str(tmp_path))
    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=DEFAULT_TASK_BINARY)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=DEFAULT_TASK_BINARY)

    result: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=DEFAULT_TASK_BINARY,
        task_type=TASK_TYPE_BINARY,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )

    row: Dict[str, Any] = result.to_row()
    _assert_metric_row_schema(row)
    assert row["metric_name"] == METRIC_MACRO_AUC
    assert 0.0 <= float(row["value"]) <= 1.0


def test_linear_probe_multiclass_subtyping_smoke_valid_metric(tmp_path: Any) -> None:
    """Multiclass smoke test: balanced accuracy is produced and bounded."""
    class_names: Tuple[str, ...] = ("A", "B", "C")
    records: List[ManifestRecord] = _make_records(
        task_name=DEFAULT_TASK_SUBTYPING,
        class_names=class_names,
        patients_per_class=5,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=DEFAULT_TASK_SUBTYPING, split_dir=str(tmp_path))
    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=DEFAULT_TASK_SUBTYPING)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=DEFAULT_TASK_SUBTYPING)

    result: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=DEFAULT_TASK_SUBTYPING,
        task_type=TASK_TYPE_SUBTYPING,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )

    row: Dict[str, Any] = result.to_row()
    _assert_metric_row_schema(row)
    assert row["metric_name"] == METRIC_BACC
    assert 0.0 <= float(row["value"]) <= 1.0


def test_linear_probe_grading_smoke_valid_metric(tmp_path: Any) -> None:
    """Grading smoke test: QWK is produced and bounded in [-1, 1]."""
    class_names: Tuple[str, ...] = ("0", "1", "2")
    records: List[ManifestRecord] = _make_records(
        task_name=DEFAULT_TASK_GRADING,
        class_names=class_names,
        patients_per_class=5,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=DEFAULT_TASK_GRADING, split_dir=str(tmp_path))
    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=DEFAULT_TASK_GRADING)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=DEFAULT_TASK_GRADING)

    result: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=DEFAULT_TASK_GRADING,
        task_type=TASK_TYPE_GRADING,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )

    row: Dict[str, Any] = result.to_row()
    _assert_metric_row_schema(row)
    assert row["metric_name"] == METRIC_QWK
    assert -1.0 <= float(row["value"]) <= 1.0


def test_linear_probe_rejects_missing_embedding_ids(tmp_path: Any) -> None:
    """Missing split IDs in embedding table should hard-fail with explicit IDs."""
    class_names: Tuple[str, ...] = ("0", "1")
    records: List[ManifestRecord] = _make_records(
        task_name=DEFAULT_TASK_BINARY,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=DEFAULT_TASK_BINARY, split_dir=str(tmp_path))
    split_bad: Dict[str, Any] = dict(split)
    split_bad["test_ids"] = list(split["test_ids"]) + ["missing_sample_id"]

    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=DEFAULT_TASK_BINARY)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=DEFAULT_TASK_BINARY)

    with pytest.raises(ValueError, match="missing_sample_id"):
        _ = _evaluate_one_fold(
            split=split_bad,
            task_name=DEFAULT_TASK_BINARY,
            task_type=TASK_TYPE_BINARY,
            unit="slide",
            id_to_embedding=embeddings,
            sample_to_patient=_sample_to_patient(records),
            id_to_label=labels,
        )


def test_linear_probe_rejects_invalid_split_schema() -> None:
    """Malformed split payload should fail fast via SplitManager validation."""
    manager: SplitManager = SplitManager(split_dir="/tmp", seed=SPLIT_DEFAULT_SEED)
    bad_split: Dict[str, Any] = {
        "task_name": DEFAULT_TASK_BINARY,
        "fold_id": "0",
        "train_ids": ["s0", "s1"],
        # Required: test_ids is intentionally missing.
    }
    with pytest.raises(SplitValidationError):
        _ = manager.make_fewshot(base_split=bad_split, k_per_class=1)


def test_linear_probe_parameter_lock() -> None:
    """Evaluator must retain fixed paper/config hyperparameters."""
    evaluator: LinearProbeEvaluator = LinearProbeEvaluator(
        c_value=DEFAULT_C_VALUE,
        solver=DEFAULT_SOLVER,
        max_iter=DEFAULT_MAX_ITER,
        class_weight=DEFAULT_CLASS_WEIGHT,
    )
    assert evaluator._c_value == DEFAULT_C_VALUE
    assert evaluator._solver == DEFAULT_SOLVER
    assert evaluator._max_iter == DEFAULT_MAX_ITER
    assert evaluator._class_weight == DEFAULT_CLASS_WEIGHT


def test_linear_probe_deterministic_repeated_run(tmp_path: Any) -> None:
    """Repeated same-input evaluation should produce identical outputs."""
    class_names: Tuple[str, ...] = ("0", "1")
    records: List[ManifestRecord] = _make_records(
        task_name=DEFAULT_TASK_BINARY,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=DEFAULT_TASK_BINARY, split_dir=str(tmp_path))
    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=DEFAULT_TASK_BINARY)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=DEFAULT_TASK_BINARY)

    first: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=DEFAULT_TASK_BINARY,
        task_type=TASK_TYPE_BINARY,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )
    second: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=DEFAULT_TASK_BINARY,
        task_type=TASK_TYPE_BINARY,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )

    assert first.metric_name == second.metric_name
    assert first.task == second.task
    assert first.fold == second.fold
    assert first.model_name == second.model_name
    assert first.value == pytest.approx(second.value, abs=1.0e-12)


def test_linear_probe_deterministic_with_regenerated_splits(tmp_path: Any) -> None:
    """Regenerated CV splits from same seed must yield identical fold metrics."""
    class_names: Tuple[str, ...] = ("0", "1")
    task_name: str = DEFAULT_TASK_BINARY
    records: List[ManifestRecord] = _make_records(
        task_name=task_name,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    manager_a: SplitManager = SplitManager(split_dir=str(tmp_path / "a"), seed=SPLIT_DEFAULT_SEED)
    manager_b: SplitManager = SplitManager(split_dir=str(tmp_path / "b"), seed=SPLIT_DEFAULT_SEED)

    splits_a: List[Dict[str, Any]] = manager_a.make_cv(
        records=records,
        n_folds=DEFAULT_CV_FOLDS,
        stratify_by=task_name,
        group_by=DEFAULT_GROUP_BY,
    )
    splits_b: List[Dict[str, Any]] = manager_b.make_cv(
        records=records,
        n_folds=DEFAULT_CV_FOLDS,
        stratify_by=task_name,
        group_by=DEFAULT_GROUP_BY,
    )

    assert _canonicalize_splits(splits_a) == _canonicalize_splits(splits_b)

    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=task_name)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=task_name)
    sample_to_patient: Dict[str, str] = _sample_to_patient(records)

    values_a: List[float] = []
    values_b: List[float] = []
    for split_a, split_b in zip(splits_a, splits_b):
        result_a: _EvalResult = _evaluate_one_fold(
            split=split_a,
            task_name=task_name,
            task_type=TASK_TYPE_BINARY,
            unit="slide",
            id_to_embedding=embeddings,
            sample_to_patient=sample_to_patient,
            id_to_label=labels,
        )
        result_b: _EvalResult = _evaluate_one_fold(
            split=split_b,
            task_name=task_name,
            task_type=TASK_TYPE_BINARY,
            unit="slide",
            id_to_embedding=embeddings,
            sample_to_patient=sample_to_patient,
            id_to_label=labels,
        )
        values_a.append(result_a.value)
        values_b.append(result_b.value)

    assert values_a == pytest.approx(values_b, abs=1.0e-12)


def test_linear_probe_fewshot_test_set_unchanged(tmp_path: Any) -> None:
    """Few-shot split must keep test IDs fixed and yield valid metric."""
    class_names: Tuple[str, ...] = ("0", "1")
    task_name: str = DEFAULT_TASK_BINARY
    records: List[ManifestRecord] = _make_records(
        task_name=task_name,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=DEFAULT_SLIDES_PER_PATIENT,
    )
    manager: SplitManager = SplitManager(split_dir=str(tmp_path), seed=SPLIT_DEFAULT_SEED)
    base_split: Dict[str, Any] = manager.make_cv(
        records=records,
        n_folds=DEFAULT_CV_FOLDS,
        stratify_by=task_name,
        group_by=DEFAULT_GROUP_BY,
    )[0]
    fewshot: Dict[str, Any] = manager.make_fewshot(base_split=base_split, k_per_class=1)

    assert list(fewshot["test_ids"]) == list(base_split["test_ids"])

    embeddings: Dict[str, np.ndarray] = _build_slide_embeddings(records=records, task_name=task_name)
    labels: Dict[str, str] = _build_slide_label_map(records=records, task_name=task_name)

    result: _EvalResult = _evaluate_one_fold(
        split=fewshot,
        task_name=task_name,
        task_type=TASK_TYPE_BINARY,
        unit="slide",
        id_to_embedding=embeddings,
        sample_to_patient=_sample_to_patient(records),
        id_to_label=labels,
    )
    assert result.metric_name == METRIC_MACRO_AUC
    assert 0.0 <= float(result.value) <= 1.0


def test_linear_probe_patient_group_no_leakage(tmp_path: Any) -> None:
    """Patient-level run must preserve patient-disjoint train/test groups."""
    class_names: Tuple[str, ...] = ("0", "1")
    task_name: str = DEFAULT_TASK_BINARY
    records: List[ManifestRecord] = _make_records(
        task_name=task_name,
        class_names=class_names,
        patients_per_class=6,
        slides_per_patient=3,
    )
    split: Dict[str, Any] = _make_cv_first_split(records=records, task_name=task_name, split_dir=str(tmp_path))
    sample_to_patient: Dict[str, str] = _sample_to_patient(records)
    _assert_no_patient_overlap(split=split, sample_to_patient=sample_to_patient)

    patient_embeddings: Dict[str, np.ndarray] = _build_patient_embeddings(records=records, task_name=task_name)
    patient_labels: Dict[str, str] = _build_patient_label_map(records=records, task_name=task_name)

    result: _EvalResult = _evaluate_one_fold(
        split=split,
        task_name=task_name,
        task_type=TASK_TYPE_BINARY,
        unit="patient",
        id_to_embedding=patient_embeddings,
        sample_to_patient=sample_to_patient,
        id_to_label=patient_labels,
    )

    assert result.metric_name == METRIC_MACRO_AUC
    assert 0.0 <= float(result.value) <= 1.0


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _make_records(
    task_name: str,
    class_names: Sequence[str],
    patients_per_class: int,
    slides_per_patient: int,
) -> List[ManifestRecord]:
    records: List[ManifestRecord] = []
    patient_counter: int = 0

    for class_name in class_names:
        for _ in range(patients_per_class):
            patient_id: str = f"patient_{patient_counter:03d}"
            for slide_idx in range(slides_per_patient):
                sample_id: str = f"sample_{patient_counter:03d}_{slide_idx:02d}"
                records.append(
                    ManifestRecord(
                        sample_id=sample_id,
                        patient_id=patient_id,
                        cohort=DEFAULT_COHORT,
                        slide_path=f"/tmp/{sample_id}.svs",
                        magnification=20,
                        rna_path="",
                        dna_path="",
                        task_labels={task_name: str(class_name)},
                        meta={},
                    )
                )
            patient_counter += 1

    return records


def _make_cv_first_split(records: Sequence[ManifestRecord], task_name: str, split_dir: str) -> Dict[str, Any]:
    manager: SplitManager = SplitManager(split_dir=split_dir, seed=SPLIT_DEFAULT_SEED)
    splits: List[Dict[str, Any]] = manager.make_cv(
        records=list(records),
        n_folds=DEFAULT_CV_FOLDS,
        stratify_by=task_name,
        group_by=DEFAULT_GROUP_BY,
    )
    return splits[0]


def _sample_to_patient(records: Sequence[ManifestRecord]) -> Dict[str, str]:
    return {record.sample_id: record.patient_id for record in records}


def _build_slide_label_map(records: Sequence[ManifestRecord], task_name: str) -> Dict[str, str]:
    return {
        record.sample_id: str(record.task_labels[task_name])
        for record in records
    }


def _build_patient_label_map(records: Sequence[ManifestRecord], task_name: str) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for record in records:
        patient_id: str = record.patient_id
        label_value: str = str(record.task_labels[task_name])
        if patient_id not in output:
            output[patient_id] = label_value
            continue
        if output[patient_id] != label_value:
            raise AssertionError(
                f"Inconsistent labels for patient_id={patient_id}: {output[patient_id]} vs {label_value}"
            )
    return output


def _build_slide_embeddings(records: Sequence[ManifestRecord], task_name: str) -> Dict[str, np.ndarray]:
    class_to_index: Dict[str, int] = _class_to_index(records=records, task_name=task_name)
    output: Dict[str, np.ndarray] = {}

    for entity_index, record in enumerate(sorted(records, key=lambda r: r.sample_id)):
        class_index: int = class_to_index[str(record.task_labels[task_name])]
        output[record.sample_id] = _make_embedding_vector(class_index=class_index, entity_index=entity_index)
    return output


def _build_patient_embeddings(records: Sequence[ManifestRecord], task_name: str) -> Dict[str, np.ndarray]:
    class_to_index: Dict[str, int] = _class_to_index(records=records, task_name=task_name)
    patient_label_map: Dict[str, str] = _build_patient_label_map(records=records, task_name=task_name)

    output: Dict[str, np.ndarray] = {}
    for entity_index, patient_id in enumerate(sorted(patient_label_map.keys())):
        class_index: int = class_to_index[patient_label_map[patient_id]]
        output[patient_id] = _make_embedding_vector(class_index=class_index, entity_index=entity_index)
    return output


def _class_to_index(records: Sequence[ManifestRecord], task_name: str) -> Dict[str, int]:
    classes: List[str] = sorted({str(record.task_labels[task_name]) for record in records})
    return {class_name: idx for idx, class_name in enumerate(classes)}


def _make_embedding_vector(class_index: int, entity_index: int) -> np.ndarray:
    vector_seed: int = DEFAULT_SEED + (class_index * 104729) + (entity_index * 1009)
    rng: np.random.Generator = np.random.default_rng(vector_seed)

    vector: np.ndarray = rng.normal(loc=0.0, scale=0.05, size=DEFAULT_EMBEDDING_DIM).astype(np.float64)
    vector[class_index % 64] += 4.0
    vector[(class_index + 17) % 64] += 2.0
    return vector


def _evaluate_one_fold(
    split: Mapping[str, Any],
    task_name: str,
    task_type: str,
    unit: str,
    id_to_embedding: Mapping[str, np.ndarray],
    sample_to_patient: Mapping[str, str],
    id_to_label: Mapping[str, str],
) -> _EvalResult:
    _assert_required_split_keys(split)

    raw_train_ids: List[str] = [str(item) for item in split["train_ids"]]
    raw_test_ids: List[str] = [str(item) for item in split["test_ids"]]

    train_ids: List[str] = _resolve_eval_ids(
        raw_ids=raw_train_ids,
        unit=unit,
        sample_to_patient=sample_to_patient,
    )
    test_ids: List[str] = _resolve_eval_ids(
        raw_ids=raw_test_ids,
        unit=unit,
        sample_to_patient=sample_to_patient,
    )

    x_train: np.ndarray = _stack_embeddings(ids=train_ids, id_to_embedding=id_to_embedding)
    x_test: np.ndarray = _stack_embeddings(ids=test_ids, id_to_embedding=id_to_embedding)

    y_train: np.ndarray = _collect_labels(ids=train_ids, id_to_label=id_to_label)
    y_test: np.ndarray = _collect_labels(ids=test_ids, id_to_label=id_to_label)

    evaluator: LinearProbeEvaluator = LinearProbeEvaluator(
        c_value=DEFAULT_C_VALUE,
        max_iter=DEFAULT_MAX_ITER,
        solver=DEFAULT_SOLVER,
        class_weight=DEFAULT_CLASS_WEIGHT,
    )
    evaluator.fit(x_train=x_train, y_train=y_train)
    probability: np.ndarray = np.asarray(evaluator.predict_proba(x_test), dtype=np.float64)

    if task_type == TASK_TYPE_BINARY:
        metric_name: str = METRIC_MACRO_AUC
        value: float = float(evaluator.score_binary_auc(y_true=y_test, y_prob=probability))
    else:
        class_labels: np.ndarray = np.asarray(np.unique(y_train), dtype=object)
        predicted_indices: np.ndarray = np.argmax(probability, axis=1).astype(np.int64, copy=False)
        y_pred: np.ndarray = class_labels[predicted_indices]

        if task_type == TASK_TYPE_SUBTYPING:
            metric_name = METRIC_BACC
            value = float(evaluator.score_multiclass_bacc(y_true=y_test, y_pred=y_pred))
        elif task_type == TASK_TYPE_GRADING:
            metric_name = METRIC_QWK
            value = float(evaluator.score_qwk(y_true=y_test, y_pred=y_pred))
        else:
            raise ValueError(f"Unsupported task_type: {task_type}")

    if not np.isfinite(value):
        raise ValueError(f"Metric is non-finite for task={task_name}, fold={split['fold_id']}.")

    return _EvalResult(
        fold=str(split["fold_id"]),
        metric_name=metric_name,
        value=float(value),
        task=task_name,
        model_name="THREADS",
    )


def _resolve_eval_ids(
    raw_ids: Sequence[str],
    unit: str,
    sample_to_patient: Mapping[str, str],
) -> List[str]:
    if unit == "slide":
        return sorted({sample_id.strip() for sample_id in raw_ids if sample_id.strip() != ""})

    if unit == "patient":
        output: List[str] = []
        for sample_id_raw in raw_ids:
            sample_id: str = sample_id_raw.strip()
            if sample_id == "":
                continue
            if sample_id not in sample_to_patient:
                continue
            output.append(sample_to_patient[sample_id])
        return sorted(set(output))

    raise ValueError(f"Unsupported evaluation unit: {unit}")


def _stack_embeddings(ids: Sequence[str], id_to_embedding: Mapping[str, np.ndarray]) -> np.ndarray:
    vectors: List[np.ndarray] = []
    missing_ids: List[str] = []

    for item_id in ids:
        if item_id not in id_to_embedding:
            missing_ids.append(item_id)
            continue
        vector: np.ndarray = np.asarray(id_to_embedding[item_id], dtype=np.float64).reshape(-1)
        if int(vector.shape[0]) != DEFAULT_EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch for id={item_id}; "
                f"expected={DEFAULT_EMBEDDING_DIM}, got={int(vector.shape[0])}."
            )
        if not np.isfinite(vector).all():
            raise ValueError(f"Embedding contains NaN/Inf for id={item_id}.")
        vectors.append(vector)

    if len(missing_ids) > 0:
        raise ValueError(
            "Missing embeddings for required IDs: "
            f"count={len(missing_ids)} preview={missing_ids[:20]}"
        )
    if len(vectors) == 0:
        raise ValueError("No embeddings available after ID filtering.")

    return np.vstack(vectors).astype(np.float64, copy=False)


def _collect_labels(ids: Sequence[str], id_to_label: Mapping[str, str]) -> np.ndarray:
    values: List[str] = []
    missing_ids: List[str] = []

    for item_id in ids:
        if item_id not in id_to_label:
            missing_ids.append(item_id)
            continue
        label_value: str = str(id_to_label[item_id]).strip()
        if label_value == "":
            missing_ids.append(item_id)
            continue
        values.append(label_value)

    if len(missing_ids) > 0:
        raise ValueError(
            "Missing labels for required IDs: "
            f"count={len(missing_ids)} preview={missing_ids[:20]}"
        )

    return np.asarray(values, dtype=object)


def _assert_required_split_keys(split: Mapping[str, Any]) -> None:
    required: Tuple[str, ...] = ("task_name", "fold_id", "train_ids", "test_ids")
    missing: List[str] = [key for key in required if key not in split]
    if len(missing) > 0:
        raise ValueError(f"Split missing required keys: {missing}")


def _assert_metric_row_schema(row: Mapping[str, Any]) -> None:
    required: Tuple[str, ...] = ("task", "fold", "metric_name", "value", "model_name")
    missing: List[str] = [key for key in required if key not in row]
    assert len(missing) == 0

    value_float: float = float(row["value"])
    assert np.isfinite(value_float)


def _assert_no_patient_overlap(split: Mapping[str, Any], sample_to_patient: Mapping[str, str]) -> None:
    train_ids: List[str] = [str(item) for item in split["train_ids"]]
    test_ids: List[str] = [str(item) for item in split["test_ids"]]

    train_patients: set[str] = {
        sample_to_patient[sample_id]
        for sample_id in train_ids
        if sample_id in sample_to_patient
    }
    test_patients: set[str] = {
        sample_to_patient[sample_id]
        for sample_id in test_ids
        if sample_id in sample_to_patient
    }

    assert train_patients.isdisjoint(test_patients)


def _canonicalize_splits(splits: Sequence[Mapping[str, Any]]) -> Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...]:
    output: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
    for split in splits:
        fold_id: str = str(split["fold_id"])
        train_ids: Tuple[str, ...] = tuple(str(item) for item in split["train_ids"])
        test_ids: Tuple[str, ...] = tuple(str(item) for item in split["test_ids"])
        output.append((fold_id, train_ids, test_ids))
    return tuple(output)
