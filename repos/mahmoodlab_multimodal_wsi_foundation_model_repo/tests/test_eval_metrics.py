## tests/test_eval_metrics.py
"""Unit tests for evaluation metrics across TITAN evaluators.

These tests validate metric correctness and schema contracts for:
- linear probing (`src/eval/linear_probe.py`)
- zero-shot classification (`src/eval/zero_shot.py`)
- slide/cross-modal retrieval (`src/eval/retrieval.py`)
- report-generation metrics (`src/eval/report_generation.py`)
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch

from src.data.build_feature_grid import FeatureGrid
from src.data.datasets import BaseDataset
from src.eval.linear_probe import LinearProbeEvaluator
from src.eval.report_generation import ReportGenerationEvaluator
from src.eval.retrieval import RetrievalEvaluator
from src.eval.zero_shot import ZeroShotEvaluator
from src.models.coca_multimodal import CoCaModel


_FEATURE_DIM: int = 768
_PATCH_SIZE: int = 512
_MAGNIFICATION: str = "20x"
_SLIDE_K: Tuple[int, int, int] = (1, 3, 5)
_CROSS_MODAL_K: Tuple[int, int, int, int] = (1, 3, 5, 10)
_NUM_BEAMS: int = 5
_NUM_BEAM_GROUPS: int = 1


def _zeros_embedding(num_rows: int) -> np.ndarray:
    """Create deterministic embedding matrix [N, 768] with float32 dtype."""
    matrix: np.ndarray = np.zeros((num_rows, _FEATURE_DIM), dtype=np.float32)
    return matrix


def _unit_axis(axis: int) -> np.ndarray:
    """Create one 768-dim unit basis vector as float32."""
    vector: np.ndarray = np.zeros((_FEATURE_DIM,), dtype=np.float32)
    vector[axis] = 1.0
    return vector


def _make_feature_grid(slide_id: str) -> FeatureGrid:
    """Create a minimal valid FeatureGrid [1,1,768] for report-generation tests."""
    features: torch.Tensor = torch.zeros((1, 1, _FEATURE_DIM), dtype=torch.float32)
    coords_xy: torch.Tensor = torch.zeros((1, 1, 2), dtype=torch.int64)
    valid_mask: torch.Tensor = torch.ones((1, 1), dtype=torch.bool)
    return FeatureGrid(
        features=features,
        coords_xy=coords_xy,
        valid_mask=valid_mask,
        slide_id=slide_id,
    )


class _DummyReportDataset(BaseDataset):
    """Simple in-memory BaseDataset implementation for evaluator tests."""

    def __init__(self, samples: Sequence[Dict[str, Any]]) -> None:
        self._samples: List[Dict[str, Any]] = [dict(item) for item in samples]

    def __len__(self) -> int:
        return int(len(self._samples))

    def __getitem__(self, idx: int) -> dict:
        return dict(self._samples[int(idx)])


class _DummyCoCaModel(CoCaModel):
    """Minimal CoCaModel-compatible stub for report-generation tests."""

    def __init__(self, generated_reports: Sequence[str]) -> None:
        torch.nn.Module.__init__(self)
        self.embed_dim: int = _FEATURE_DIM
        self._generated_reports: List[str] = [str(item) for item in generated_reports]
        self._cursor: int = 0
        self._frozen_param: torch.nn.Parameter = torch.nn.Parameter(
            torch.zeros((1,), dtype=torch.float32),
            requires_grad=False,
        )

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        return iter([self._frozen_param])

    def generate_report(self, grid: FeatureGrid, num_beams: int = _NUM_BEAMS) -> str:  # type: ignore[override]
        if not isinstance(grid, FeatureGrid):
            raise TypeError(f"grid must be FeatureGrid, got {type(grid).__name__}.")
        if int(num_beams) != _NUM_BEAMS:
            raise ValueError(f"num_beams must be {_NUM_BEAMS}, got {num_beams}.")
        if len(self._generated_reports) == 0:
            return ""
        report: str = self._generated_reports[self._cursor % len(self._generated_reports)]
        self._cursor += 1
        return report


def test_config_locked_eval_constants_sanity() -> None:
    """Config-locked constants should match the evaluation protocol."""
    assert _PATCH_SIZE == 512
    assert _MAGNIFICATION == "20x"
    assert _FEATURE_DIM == 768
    assert _SLIDE_K == (1, 3, 5)
    assert _CROSS_MODAL_K == (1, 3, 5, 10)
    assert _NUM_BEAMS == 5
    assert _NUM_BEAM_GROUPS == 1


def test_zero_shot_binary_perfect_predictions_have_balanced_accuracy_and_auroc_one() -> None:
    """Binary zero-shot should produce perfect BA/AUROC on separable aligned embeddings."""
    evaluator: ZeroShotEvaluator = ZeroShotEvaluator()

    class_text_emb: np.ndarray = np.stack([_unit_axis(0), _unit_axis(1)], axis=0)
    y_true: np.ndarray = np.asarray([0, 1, 1, 0], dtype=np.int64)
    slide_emb: np.ndarray = np.stack(
        [class_text_emb[int(index)] for index in y_true.tolist()],
        axis=0,
    )

    output: Dict[str, Any] = evaluator.run_zero_shot(
        slide_emb=slide_emb,
        class_text_emb=class_text_emb,
        y=y_true,
    )
    metrics: Dict[str, float] = output["metrics"]

    assert metrics["balanced_accuracy"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["auroc"] == pytest.approx(1.0, abs=1e-12)
    assert "weighted_f1" not in metrics


def test_zero_shot_binary_inverted_scores_have_auroc_zero() -> None:
    """Binary zero-shot AUROC should be 0.0 when rankings are perfectly inverted."""
    evaluator: ZeroShotEvaluator = ZeroShotEvaluator()

    class_text_emb: np.ndarray = np.stack([_unit_axis(0), _unit_axis(1)], axis=0)
    y_true: np.ndarray = np.asarray([0, 0, 1, 1], dtype=np.int64)

    # Negative samples score high on class-1, positives score low on class-1.
    slide_emb: np.ndarray = np.stack(
        [_unit_axis(1), _unit_axis(1), _unit_axis(0), _unit_axis(0)],
        axis=0,
    )

    output: Dict[str, Any] = evaluator.run_zero_shot(
        slide_emb=slide_emb,
        class_text_emb=class_text_emb,
        y=y_true,
    )
    metrics: Dict[str, float] = output["metrics"]

    assert metrics["auroc"] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_zero_shot_multiclass_balanced_accuracy_matches_macro_recall() -> None:
    """Balanced accuracy should equal mean class recall on a controlled multiclass pattern."""
    evaluator: ZeroShotEvaluator = ZeroShotEvaluator()

    class_text_emb: np.ndarray = np.stack(
        [_unit_axis(0), _unit_axis(1), _unit_axis(2)],
        axis=0,
    )
    y_true: np.ndarray = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred_forced: np.ndarray = np.asarray([0, 1, 1, 1, 0, 2], dtype=np.int64)
    slide_emb: np.ndarray = np.stack(
        [class_text_emb[int(index)] for index in y_pred_forced.tolist()],
        axis=0,
    )

    output: Dict[str, Any] = evaluator.run_zero_shot(
        slide_emb=slide_emb,
        class_text_emb=class_text_emb,
        y=y_true,
    )
    metrics: Dict[str, float] = output["metrics"]

    expected_balanced_accuracy: float = (0.5 + 1.0 + 0.5) / 3.0
    assert metrics["balanced_accuracy"] == pytest.approx(expected_balanced_accuracy, abs=1e-12)
    assert "weighted_f1" in metrics
    assert "auroc" not in metrics


def test_linear_probe_output_schema_contains_required_metric_keys() -> None:
    """Linear probe output schema should expose expected fold and aggregate metric keys."""
    evaluator: LinearProbeEvaluator = LinearProbeEvaluator()

    features: np.ndarray = _zeros_embedding(12)
    labels: np.ndarray = np.asarray([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    features[:6, 0] = 1.0
    features[6:, 1] = 1.0

    split: Dict[str, Any] = {
        "train": np.asarray([0, 1, 2, 6, 7, 8], dtype=np.int64),
        "test": np.asarray([3, 4, 5, 9, 10, 11], dtype=np.int64),
    }
    output: Dict[str, Any] = evaluator.run_linear_probe(
        features=features,
        y=labels,
        split=split,
    )

    assert output["task"] == "linear_probe"
    assert "folds" in output
    assert "aggregate_test" in output

    aggregate_test: Dict[str, float] = output["aggregate_test"]
    assert "balanced_accuracy" in aggregate_test
    assert "auroc" in aggregate_test
    assert 0.0 <= float(aggregate_test["balanced_accuracy"]) <= 1.0
    assert 0.0 <= float(aggregate_test["auroc"]) <= 1.0


def test_slide_retrieval_acc_at_k_is_monotonic_and_perfect_for_exact_matches() -> None:
    """Slide retrieval Acc@K should be monotonic and perfect for exact class matches."""
    evaluator: RetrievalEvaluator = RetrievalEvaluator()

    db: np.ndarray = _zeros_embedding(6)
    ydb: np.ndarray = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    for idx in range(6):
        db[idx, idx] = 1.0

    query: np.ndarray = np.stack([db[0], db[1], db[2]], axis=0).astype(np.float32, copy=False)
    yq: np.ndarray = np.asarray([0, 1, 2], dtype=np.int64)

    output: Dict[str, Any] = evaluator.run_slide_retrieval(
        query=query,
        db=db,
        yq=yq,
        ydb=ydb,
    )
    metrics: Dict[str, float] = output["metrics"]

    acc1: float = float(metrics["acc_at_1"])
    acc3: float = float(metrics["acc_at_3"])
    acc5: float = float(metrics["acc_at_5"])
    assert 0.0 <= acc1 <= acc3 <= acc5 <= 1.0
    assert acc1 == pytest.approx(1.0, abs=1e-12)
    assert acc3 == pytest.approx(1.0, abs=1e-12)
    assert acc5 == pytest.approx(1.0, abs=1e-12)


def test_slide_retrieval_mvacc_tie_break_is_deterministic() -> None:
    """MVAcc@5 tie-break should be deterministic and follow distance-sum policy."""
    evaluator: RetrievalEvaluator = RetrievalEvaluator()

    # Keep database mean at zero so center+L2 preprocessing is stable/predictable.
    db: np.ndarray = _zeros_embedding(5)
    ydb: np.ndarray = np.asarray([0, 0, 1, 1, 2], dtype=np.int64)
    db[0, 0] = 1.0
    db[1, 1] = 1.0
    db[2, 0] = -1.0
    db[3, 1] = -1.0
    db[4, 0] = 0.0
    db[4, 1] = 0.0

    query: np.ndarray = _zeros_embedding(1)
    query[0, 0] = 1.0
    query[0, 1] = 1.0
    yq: np.ndarray = np.asarray([0], dtype=np.int64)

    output_a: Dict[str, Any] = evaluator.run_slide_retrieval(query=query, db=db, yq=yq, ydb=ydb)
    output_b: Dict[str, Any] = evaluator.run_slide_retrieval(query=query, db=db, yq=yq, ydb=ydb)

    metrics_a: Dict[str, float] = output_a["metrics"]
    metrics_b: Dict[str, float] = output_b["metrics"]

    assert float(metrics_a["mvacc_at_5"]) == pytest.approx(1.0, abs=1e-12)
    assert float(metrics_b["mvacc_at_5"]) == pytest.approx(1.0, abs=1e-12)
    assert int(output_a["diagnostics"]["mv_tie_count"]) == 1
    assert output_a["metrics"] == output_b["metrics"]


def test_cross_modal_recall_is_monotonic_and_mean_recall_is_exact_average() -> None:
    """Cross-modal Recall@K should be monotonic and mean_recall must equal exact average."""
    evaluator: RetrievalEvaluator = RetrievalEvaluator()

    num_pairs: int = 10
    labels: np.ndarray = np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    slide_emb: np.ndarray = _zeros_embedding(num_pairs)
    report_emb: np.ndarray = _zeros_embedding(num_pairs)

    for idx in range(num_pairs):
        axis: int = idx % 5
        slide_emb[idx, axis] = 1.0
        report_emb[idx, axis] = 1.0

    output: Dict[str, Any] = evaluator.run_cross_modal_retrieval(
        slide_emb=slide_emb,
        report_emb=report_emb,
        labels=labels,
    )

    for direction_key in ("slide_to_report", "report_to_slide"):
        metrics: Dict[str, float] = output[direction_key]["metrics"]
        values: List[float] = [float(metrics[f"recall_at_{k_value}"]) for k_value in _CROSS_MODAL_K]
        assert 0.0 <= values[0] <= values[1] <= values[2] <= values[3] <= 1.0
        expected_mean: float = float(np.mean(np.asarray(values, dtype=np.float64)))
        assert float(metrics["mean_recall"]) == pytest.approx(expected_mean, abs=1e-12)


def test_report_generation_metric_primitives_perfect_vs_unrelated_and_bounded() -> None:
    """METEOR/ROUGE-1/BLEU-1 should be bounded and degrade on unrelated generations."""
    evaluator: ReportGenerationEvaluator = ReportGenerationEvaluator()
    reference_text: str = "Microscopic examination confirms squamous cell carcinoma."

    perfect: Dict[str, float] = evaluator._compute_sample_metrics(  # pylint: disable=protected-access
        reference_text=reference_text,
        generated_text=reference_text,
    )
    unrelated: Dict[str, float] = evaluator._compute_sample_metrics(  # pylint: disable=protected-access
        reference_text=reference_text,
        generated_text="No diagnostic morphology is identified in this synthetic sentence.",
    )

    for metrics in (perfect, unrelated):
        assert 0.0 <= float(metrics["meteor"]) <= 1.0
        assert 0.0 <= float(metrics["rouge1"]) <= 1.0
        assert 0.0 <= float(metrics["bleu1"]) <= 1.0

    assert float(perfect["meteor"]) >= float(unrelated["meteor"])
    assert float(perfect["rouge1"]) >= float(unrelated["rouge1"])
    assert float(perfect["bleu1"]) >= float(unrelated["bleu1"])


def test_report_generation_public_api_returns_expected_schema(tmp_path: Any) -> None:
    """run_report_generation should return required protocol/aggregate/bootstrap schema."""
    references: List[str] = [
        "The slide shows invasive ductal carcinoma.",
        "The slide demonstrates papillary renal cell carcinoma.",
    ]
    dataset: _DummyReportDataset = _DummyReportDataset(
        samples=[
            {
                "sample_id": "s0",
                "slide_id": "slide_0",
                "image_grid": _make_feature_grid("slide_0"),
                "reference_report": references[0],
            },
            {
                "sample_id": "s1",
                "slide_id": "slide_1",
                "image_grid": _make_feature_grid("slide_1"),
                "reference_report": references[1],
            },
        ]
    )
    model: _DummyCoCaModel = _DummyCoCaModel(generated_reports=references)
    evaluator: ReportGenerationEvaluator = ReportGenerationEvaluator(output_dir=str(tmp_path))

    output: Dict[str, Any] = evaluator.run_report_generation(model=model, dataset=dataset)

    assert output["task"] == "report_generation"
    assert int(output["protocol"]["decoding"]["num_beams"]) == _NUM_BEAMS
    assert int(output["protocol"]["decoding"]["num_beam_groups"]) == _NUM_BEAM_GROUPS
    assert int(output["input"]["required_embedding_dim"]) == _FEATURE_DIM

    aggregate: Dict[str, float] = output["aggregate"]
    assert "meteor_mean" in aggregate
    assert "rouge1_mean" in aggregate
    assert "bleu1_mean" in aggregate
    assert 0.0 <= float(aggregate["meteor_mean"]) <= 1.0
    assert 0.0 <= float(aggregate["rouge1_mean"]) <= 1.0
    assert 0.0 <= float(aggregate["bleu1_mean"]) <= 1.0

    bootstrap: Dict[str, Any] = output["bootstrap"]
    assert int(bootstrap["n_bootstrap"]) == 1000
    assert "metrics" in bootstrap
    assert set(bootstrap["metrics"].keys()) == {"meteor", "rouge1", "bleu1"}
