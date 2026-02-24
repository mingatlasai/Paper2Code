"""Report-generation evaluation for TITAN multimodal model.

This module implements the design-locked interface:
- Evaluator.run_report_generation(model: CoCaModel, dataset: BaseDataset) -> dict

Protocol alignment (paper + config):
- Zero-shot report generation from stage-3 multimodal TITAN.
- Beam-search decoding with num_beams=5 and num_beam_groups=1.
- Metrics: METEOR, ROUGE-1, BLEU-1.
- Single-split uncertainty via bootstrap with n=1000.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from src.core.utils import ensure_dir, write_csv, write_json
from src.data.build_feature_grid import FeatureGrid
from src.data.datasets import BaseDataset, MultimodalBatch
from src.models.coca_multimodal import CoCaModel

try:
    from nltk.translate.meteor_score import meteor_score
except Exception as exc:  # pragma: no cover - dependency import contract
    raise ImportError("nltk is required for METEOR computation.") from exc

try:
    from rouge_score import rouge_scorer
except Exception as exc:  # pragma: no cover - dependency import contract
    raise ImportError("rouge-score is required for ROUGE-1 computation.") from exc

try:
    import sacrebleu
except Exception as exc:  # pragma: no cover - dependency import contract
    raise ImportError("sacrebleu is required for BLEU-1 computation.") from exc


# -----------------------------------------------------------------------------
# Config-locked constants from provided configuration.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DECODING_STRATEGY: str = "beam_search"
_NUM_BEAMS: int = 5
_NUM_BEAM_GROUPS: int = 1

_BOOTSTRAP_SAMPLES: int = 1000
_BOOTSTRAP_CI_ALPHA: float = 0.05
_BOOTSTRAP_SEED: int = 42

_NORMALIZE_WHITESPACE: bool = True
_LOWERCASE_TEXT: bool = False
_STRIP_TEXT: bool = True

_WRITE_JSON: bool = True
_WRITE_CSV: bool = True
_OUTPUT_DIR: str = "./outputs/eval/report_generation"
_PREDICTIONS_FILENAME: str = "report_generation_predictions.csv"
_PER_SAMPLE_METRICS_FILENAME: str = "report_generation_per_sample_metrics.csv"
_SUMMARY_FILENAME: str = "report_generation_summary.csv"
_METRICS_FILENAME: str = "report_generation_metrics.json"

_REQUIRE_EMBEDDING_DIM_MATCH: bool = True
_REQUIRED_EMBEDDING_DIM: int = 768
_REQUIRE_STAGE3_MULTIMODAL_CHECKPOINT: bool = True
_REQUIRE_FROZEN_MODEL: bool = True
_REQUIRE_SLIDE_REPORT_PAIR_ALIGNMENT: bool = True
_REQUIRE_DETERMINISTIC_EVAL: bool = True

_DEFAULT_EPS: float = 1.0e-12


class ReportGenerationError(RuntimeError):
    """Base exception for report-generation evaluation failures."""


class ReportGenerationSchemaError(ReportGenerationError):
    """Raised when schema/shape contracts are violated."""


@dataclass(frozen=True)
class _SampleRecord:
    """Prepared sample for report-generation evaluation."""

    sample_id: str
    slide_id: str
    image_grid: FeatureGrid
    reference_report: str


@dataclass(frozen=True)
class _SampleResult:
    """Per-sample generation and metric result."""

    sample_id: str
    slide_id: str
    reference_report: str
    generated_report: str
    meteor: float
    rouge1: float
    bleu1: float


class ReportGenerationEvaluator:
    """Evaluator for zero-shot pathology report generation."""

    def __init__(
        self,
        output_dir: str = _OUTPUT_DIR,
        write_json_enabled: bool = _WRITE_JSON,
        write_csv_enabled: bool = _WRITE_CSV,
        bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
        bootstrap_seed: int = _BOOTSTRAP_SEED,
    ) -> None:
        if not isinstance(output_dir, str) or not output_dir.strip():
            raise ValueError("output_dir must be a non-empty string.")
        if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
            raise TypeError("bootstrap_samples must be an integer.")
        if bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be > 0.")
        if int(bootstrap_samples) != _BOOTSTRAP_SAMPLES:
            raise ValueError(
                f"bootstrap_samples must be {_BOOTSTRAP_SAMPLES} per config, got {bootstrap_samples}."
            )
        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer.")
        if bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be >= 0.")

        self.output_dir: Path = ensure_dir(output_dir)
        self.write_json_enabled: bool = bool(write_json_enabled)
        self.write_csv_enabled: bool = bool(write_csv_enabled)
        self.bootstrap_samples: int = int(bootstrap_samples)
        self.bootstrap_seed: int = int(bootstrap_seed)

        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

        self.decoding_strategy: str = _DECODING_STRATEGY
        self.num_beams: int = _NUM_BEAMS
        self.num_beam_groups: int = _NUM_BEAM_GROUPS

        self.normalize_whitespace: bool = _NORMALIZE_WHITESPACE
        self.lowercase_text: bool = _LOWERCASE_TEXT
        self.strip_text: bool = _STRIP_TEXT

        self._rouge_scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

    def run_report_generation(self, model: CoCaModel, dataset: BaseDataset) -> dict:
        """Run zero-shot report generation and return structured metrics.

        Args:
            model: Multimodal CoCaModel with report generation path.
            dataset: Slide-report paired dataset.

        Returns:
            Dictionary containing protocol metadata, aggregate metrics,
            bootstrap uncertainty, per-sample counts, and artifact paths.
        """
        self._validate_runtime_constraints(model=model, dataset=dataset)

        records: List[_SampleRecord] = self._collect_samples(dataset=dataset)
        if len(records) == 0:
            raise ReportGenerationSchemaError("Dataset yielded zero valid samples.")

        model.eval()
        sample_results: List[_SampleResult] = []
        with torch.inference_mode():
            for record in records:
                generated_raw: str = self._generate_single_report(model=model, image_grid=record.image_grid)
                generated_text: str = self._normalize_text(generated_raw)
                reference_text: str = self._normalize_text(record.reference_report)

                if _REQUIRE_SLIDE_REPORT_PAIR_ALIGNMENT and len(reference_text) == 0:
                    raise ReportGenerationSchemaError(
                        f"Empty reference report after normalization for slide_id={record.slide_id}."
                    )

                metric_row: Dict[str, float] = self._compute_sample_metrics(
                    reference_text=reference_text,
                    generated_text=generated_text,
                )

                sample_results.append(
                    _SampleResult(
                        sample_id=record.sample_id,
                        slide_id=record.slide_id,
                        reference_report=reference_text,
                        generated_report=generated_text,
                        meteor=float(metric_row["meteor"]),
                        rouge1=float(metric_row["rouge1"]),
                        bleu1=float(metric_row["bleu1"]),
                    )
                )

        aggregate: Dict[str, float] = self._aggregate_metrics(sample_results)
        bootstrap: Dict[str, Any] = self._bootstrap_metrics(sample_results)

        artifacts: Dict[str, str] = self._write_artifacts(
            sample_results=sample_results,
            aggregate=aggregate,
            bootstrap=bootstrap,
        )

        output: Dict[str, Any] = {
            "task": "report_generation",
            "input": {
                "num_samples": int(len(sample_results)),
                "dataset_type": type(dataset).__name__,
                "model_type": type(model).__name__,
                "required_embedding_dim": int(_REQUIRED_EMBEDDING_DIM),
            },
            "protocol": {
                "method": "zero_shot_report_generation",
                "decoding": {
                    "strategy": self.decoding_strategy,
                    "num_beams": int(self.num_beams),
                    "num_beam_groups": int(self.num_beam_groups),
                },
                "preprocessing": {
                    "normalize_whitespace": bool(self.normalize_whitespace),
                    "lowercase_text": bool(self.lowercase_text),
                    "strip_text": bool(self.strip_text),
                },
                "metrics": ["meteor", "rouge1", "bleu1"],
                "bootstrap": {
                    "enabled": True,
                    "n_bootstrap": int(self.bootstrap_samples),
                    "seed": int(self.bootstrap_seed),
                    "ci_alpha": float(_BOOTSTRAP_CI_ALPHA),
                },
            },
            "aggregate": aggregate,
            "bootstrap": bootstrap,
            "artifacts": artifacts,
        }
        return output

    def _validate_runtime_constraints(self, model: CoCaModel, dataset: BaseDataset) -> None:
        if not isinstance(model, CoCaModel):
            raise TypeError(f"model must be CoCaModel, got {type(model).__name__}.")
        if not isinstance(dataset, BaseDataset):
            raise TypeError(f"dataset must be BaseDataset, got {type(dataset).__name__}.")

        dataset_len: int = int(len(dataset))
        if dataset_len <= 0:
            raise ReportGenerationSchemaError("dataset must contain at least one sample.")

        if _REQUIRE_EMBEDDING_DIM_MATCH:
            model_embed_dim: int = int(getattr(model, "embed_dim", -1))
            if model_embed_dim != int(_REQUIRED_EMBEDDING_DIM):
                raise ReportGenerationSchemaError(
                    "Model embedding dimension mismatch. "
                    f"Expected {_REQUIRED_EMBEDDING_DIM}, got {model_embed_dim}."
                )

        if _REQUIRE_FROZEN_MODEL:
            trainable_params: int = 0
            for parameter in model.parameters():
                if bool(parameter.requires_grad):
                    trainable_params += 1
            if trainable_params > 0:
                raise ReportGenerationSchemaError(
                    "Model must be frozen for evaluation, but trainable parameters were found."
                )

        if _REQUIRE_STAGE3_MULTIMODAL_CHECKPOINT:
            # The concrete checkpoint provenance is not represented as a formal field.
            # We enforce language capability through required public interface.
            generate_fn: Any = getattr(model, "generate_report", None)
            if not callable(generate_fn):
                raise ReportGenerationSchemaError(
                    "Model must provide generate_report for stage-3 multimodal evaluation."
                )

        if _REQUIRE_DETERMINISTIC_EVAL:
            # Deterministic protocol check: fixed decoding mode + fixed beams.
            if self.decoding_strategy != "beam_search":
                raise ReportGenerationSchemaError(
                    f"Decoding strategy must be 'beam_search', got {self.decoding_strategy}."
                )
            if self.num_beams != _NUM_BEAMS:
                raise ReportGenerationSchemaError(
                    f"num_beams must be {_NUM_BEAMS}, got {self.num_beams}."
                )
            if self.num_beam_groups != _NUM_BEAM_GROUPS:
                raise ReportGenerationSchemaError(
                    f"num_beam_groups must be {_NUM_BEAM_GROUPS}, got {self.num_beam_groups}."
                )

    def _collect_samples(self, dataset: BaseDataset) -> List[_SampleRecord]:
        output: List[_SampleRecord] = []
        dataset_len: int = int(len(dataset))

        for index in range(dataset_len):
            sample_obj: Any = dataset[index]
            record: _SampleRecord = self._parse_dataset_sample(sample_obj=sample_obj, index=index)
            output.append(record)

        return output

    def _parse_dataset_sample(self, sample_obj: Any, index: int) -> _SampleRecord:
        if isinstance(sample_obj, MultimodalBatch):
            sample_id: str = f"sample_{index}"
            slide_id: str = str(sample_obj.slide_id)
            image_grid: FeatureGrid = sample_obj.image_grid
            reference_report: str = self._decode_reference_from_tokens(
                labels=sample_obj.labels,
                input_ids=sample_obj.input_ids,
            )
            return _SampleRecord(
                sample_id=sample_id,
                slide_id=slide_id,
                image_grid=image_grid,
                reference_report=reference_report,
            )

        if isinstance(sample_obj, Mapping):
            mapping: Mapping[str, Any] = sample_obj

            image_grid_obj: Any = mapping.get("image_grid", mapping.get("grid"))
            if not isinstance(image_grid_obj, FeatureGrid):
                raise ReportGenerationSchemaError(
                    f"Sample {index} is missing FeatureGrid under 'image_grid' or 'grid'."
                )

            slide_id_obj: Any = mapping.get("slide_id", mapping.get("wsi_id", mapping.get("id", "")))
            slide_id: str = str(slide_id_obj) if str(slide_id_obj) else f"slide_{index}"

            sample_id_obj: Any = mapping.get("sample_id", mapping.get("pair_id", mapping.get("id", "")))
            sample_id: str = str(sample_id_obj) if str(sample_id_obj) else f"sample_{index}"

            reference_report: Optional[str] = self._extract_reference_text_from_mapping(mapping)
            if reference_report is None:
                labels_obj: Any = mapping.get("labels")
                input_ids_obj: Any = mapping.get("input_ids")
                reference_report = self._decode_reference_from_tokens(labels=labels_obj, input_ids=input_ids_obj)

            return _SampleRecord(
                sample_id=sample_id,
                slide_id=slide_id,
                image_grid=image_grid_obj,
                reference_report=reference_report,
            )

        raise ReportGenerationSchemaError(
            f"Unsupported dataset sample type at index {index}: {type(sample_obj).__name__}."
        )

    @staticmethod
    def _extract_reference_text_from_mapping(mapping: Mapping[str, Any]) -> Optional[str]:
        candidate_keys: Tuple[str, ...] = (
            "reference_report",
            "report",
            "reference_text",
            "text",
            "caption",
            "target_text",
        )
        for key in candidate_keys:
            if key in mapping:
                value: Any = mapping[key]
                if value is None:
                    continue
                text_value: str = str(value)
                return text_value
        return None

    @staticmethod
    def _decode_reference_from_tokens(labels: Any, input_ids: Any) -> str:
        token_values: Optional[List[int]] = None

        labels_tensor: Optional[torch.Tensor] = labels if isinstance(labels, torch.Tensor) else None
        input_ids_tensor: Optional[torch.Tensor] = input_ids if isinstance(input_ids, torch.Tensor) else None

        if labels_tensor is not None:
            values: List[int] = [int(v) for v in labels_tensor.detach().cpu().reshape(-1).tolist()]
            token_values = values
        elif input_ids_tensor is not None:
            values = [int(v) for v in input_ids_tensor.detach().cpu().reshape(-1).tolist()]
            token_values = values

        if token_values is None:
            raise ReportGenerationSchemaError(
                "Sample does not expose reference text and does not contain token labels/input_ids."
            )

        # Fallback token-id decoder consistent with CoCaModel implementation.
        visible_tokens: List[str] = []
        for token_id in token_values:
            token_int: int = int(token_id)
            if token_int in {0, 2}:
                continue
            if token_int == 3:
                break
            if token_int < 0:
                # Ignore label ignore indices (for example -100) if present.
                continue
            visible_tokens.append(f"tok_{token_int}")

        if len(visible_tokens) == 0:
            return ""
        return " ".join(visible_tokens)

    def _generate_single_report(self, model: CoCaModel, image_grid: FeatureGrid) -> str:
        if not isinstance(image_grid, FeatureGrid):
            raise TypeError(f"image_grid must be FeatureGrid, got {type(image_grid).__name__}.")

        model_device: torch.device = self._resolve_model_device(model)
        grid_on_device: FeatureGrid = image_grid.to(str(model_device))

        generated: str = model.generate_report(grid=grid_on_device, num_beams=self.num_beams)
        if not isinstance(generated, str):
            raise ReportGenerationSchemaError(
                f"model.generate_report must return str, got {type(generated).__name__}."
            )
        return generated

    @staticmethod
    def _resolve_model_device(model: CoCaModel) -> torch.device:
        try:
            first_param: torch.nn.Parameter = next(model.parameters())
            return first_param.device
        except StopIteration:
            return torch.device("cpu")

    def _normalize_text(self, text: str) -> str:
        normalized: str = str(text)
        if self.strip_text:
            normalized = normalized.strip()
        if self.normalize_whitespace:
            normalized = " ".join(normalized.split())
        if self.lowercase_text:
            normalized = normalized.lower()
        return normalized

    def _compute_sample_metrics(self, reference_text: str, generated_text: str) -> Dict[str, float]:
        ref_tokens: List[str] = reference_text.split()
        gen_tokens: List[str] = generated_text.split()

        meteor_value: float = float(meteor_score([ref_tokens], gen_tokens))

        rouge_scores = self._rouge_scorer.score(reference_text, generated_text)
        rouge1_value: float = float(rouge_scores["rouge1"].fmeasure)

        bleu_output = sacrebleu.sentence_bleu(
            hypothesis=generated_text,
            references=[reference_text],
            smooth_method="exp",
            use_effective_order=True,
        )
        bleu1_value: float = float(bleu_output.score) / 100.0

        return {
            "meteor": meteor_value,
            "rouge1": rouge1_value,
            "bleu1": bleu1_value,
        }

    @staticmethod
    def _aggregate_metrics(sample_results: Sequence[_SampleResult]) -> Dict[str, float]:
        if len(sample_results) == 0:
            raise ReportGenerationSchemaError("Cannot aggregate empty sample_results.")

        meteor_values: np.ndarray = np.asarray([item.meteor for item in sample_results], dtype=np.float64)
        rouge1_values: np.ndarray = np.asarray([item.rouge1 for item in sample_results], dtype=np.float64)
        bleu1_values: np.ndarray = np.asarray([item.bleu1 for item in sample_results], dtype=np.float64)

        return {
            "meteor_mean": float(np.mean(meteor_values)),
            "rouge1_mean": float(np.mean(rouge1_values)),
            "bleu1_mean": float(np.mean(bleu1_values)),
        }

    def _bootstrap_metrics(self, sample_results: Sequence[_SampleResult]) -> Dict[str, Any]:
        if len(sample_results) == 0:
            raise ReportGenerationSchemaError("Cannot bootstrap empty sample_results.")

        meteor_values: np.ndarray = np.asarray([item.meteor for item in sample_results], dtype=np.float64)
        rouge1_values: np.ndarray = np.asarray([item.rouge1 for item in sample_results], dtype=np.float64)
        bleu1_values: np.ndarray = np.asarray([item.bleu1 for item in sample_results], dtype=np.float64)

        metrics: Dict[str, np.ndarray] = {
            "meteor": meteor_values,
            "rouge1": rouge1_values,
            "bleu1": bleu1_values,
        }

        rng: np.random.Generator = np.random.default_rng(self.bootstrap_seed)
        n_samples: int = int(meteor_values.shape[0])

        output: Dict[str, Any] = {
            "n_bootstrap": int(self.bootstrap_samples),
            "seed": int(self.bootstrap_seed),
            "ci_alpha": float(_BOOTSTRAP_CI_ALPHA),
            "metrics": {},
        }

        for metric_name, values in metrics.items():
            if values.shape[0] != n_samples:
                raise ReportGenerationSchemaError(
                    f"Metric length mismatch in bootstrap for {metric_name}."
                )

            boot_means: np.ndarray = np.zeros((self.bootstrap_samples,), dtype=np.float64)
            for boot_idx in range(self.bootstrap_samples):
                sample_idx: np.ndarray = rng.integers(0, n_samples, size=n_samples, endpoint=False)
                boot_means[boot_idx] = float(np.mean(values[sample_idx]))

            lower_q: float = 100.0 * (_BOOTSTRAP_CI_ALPHA / 2.0)
            upper_q: float = 100.0 * (1.0 - (_BOOTSTRAP_CI_ALPHA / 2.0))

            metric_summary: Dict[str, float] = {
                "point_estimate": float(np.mean(values)),
                "boot_mean": float(np.mean(boot_means)),
                "boot_std": float(np.std(boot_means, ddof=1) if self.bootstrap_samples > 1 else 0.0),
                "ci_lower": float(np.percentile(boot_means, lower_q)),
                "ci_upper": float(np.percentile(boot_means, upper_q)),
            }
            output["metrics"][metric_name] = metric_summary

        return output

    def _write_artifacts(
        self,
        sample_results: Sequence[_SampleResult],
        aggregate: Mapping[str, float],
        bootstrap: Mapping[str, Any],
    ) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}

        predictions_rows: List[Dict[str, Any]] = []
        per_sample_metric_rows: List[Dict[str, Any]] = []
        for item in sample_results:
            predictions_rows.append(
                {
                    "sample_id": str(item.sample_id),
                    "slide_id": str(item.slide_id),
                    "reference_report": str(item.reference_report),
                    "generated_report": str(item.generated_report),
                }
            )
            per_sample_metric_rows.append(
                {
                    "sample_id": str(item.sample_id),
                    "slide_id": str(item.slide_id),
                    "meteor": float(item.meteor),
                    "rouge1": float(item.rouge1),
                    "bleu1": float(item.bleu1),
                }
            )

        summary_row: Dict[str, Any] = {
            "num_samples": int(len(sample_results)),
            "decoding_strategy": str(self.decoding_strategy),
            "num_beams": int(self.num_beams),
            "num_beam_groups": int(self.num_beam_groups),
            "meteor_mean": float(aggregate["meteor_mean"]),
            "rouge1_mean": float(aggregate["rouge1_mean"]),
            "bleu1_mean": float(aggregate["bleu1_mean"]),
        }

        if self.write_csv_enabled:
            predictions_path: Path = write_csv(
                predictions_rows,
                self.output_dir / _PREDICTIONS_FILENAME,
            )
            per_sample_path: Path = write_csv(
                per_sample_metric_rows,
                self.output_dir / _PER_SAMPLE_METRICS_FILENAME,
            )
            summary_path: Path = write_csv(
                [summary_row],
                self.output_dir / _SUMMARY_FILENAME,
            )
            artifacts["predictions_csv"] = str(predictions_path)
            artifacts["per_sample_metrics_csv"] = str(per_sample_path)
            artifacts["summary_csv"] = str(summary_path)

        if self.write_json_enabled:
            payload: Dict[str, Any] = {
                "task": "report_generation",
                "protocol": {
                    "method": "zero_shot_report_generation",
                    "decoding": {
                        "strategy": str(self.decoding_strategy),
                        "num_beams": int(self.num_beams),
                        "num_beam_groups": int(self.num_beam_groups),
                    },
                    "text_normalization": {
                        "normalize_whitespace": bool(self.normalize_whitespace),
                        "lowercase_text": bool(self.lowercase_text),
                        "strip_text": bool(self.strip_text),
                    },
                },
                "aggregate": dict(aggregate),
                "bootstrap": dict(bootstrap),
            }
            metrics_path: Path = write_json(
                payload,
                self.output_dir / _METRICS_FILENAME,
            )
            artifacts["metrics_json"] = str(metrics_path)

        return artifacts


class Evaluator(ReportGenerationEvaluator):
    """Design-compat alias exposing required method name and signature."""


# Convenience functional API.
def run_report_generation(model: CoCaModel, dataset: BaseDataset) -> dict:
    """Run report-generation evaluation with config-aligned defaults."""
    evaluator: ReportGenerationEvaluator = ReportGenerationEvaluator()
    return evaluator.run_report_generation(model=model, dataset=dataset)


__all__ = [
    "ReportGenerationError",
    "ReportGenerationSchemaError",
    "ReportGenerationEvaluator",
    "Evaluator",
    "run_report_generation",
]
