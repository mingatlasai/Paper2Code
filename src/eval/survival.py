"""Survival evaluation for TITAN frozen slide embeddings.

This module implements the design-locked interface:
- Evaluator.run_survival(features, time, event, folds) -> dict

Protocol alignment (paper + config):
- Model: linear Cox proportional hazards (`scikit-survival`).
- Endpoint: disease-specific survival.
- Primary metric: concordance index (c-index).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored
except Exception as exc:  # pragma: no cover - import contract
    raise ImportError("scikit-survival is required for survival evaluation.") from exc


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml + configs/eval/survival.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_TASK_NAME: str = "survival"
_ENDPOINT_NAME: str = "disease_specific_survival"

_MODEL_NAME: str = "linear Cox proportional hazards"
_MODEL_PACKAGE: str = "scikit-survival"

_CENTER_EMBEDDINGS: bool = False
_L2_NORMALIZE_EMBEDDINGS: bool = False
_FIT_STATS_ON_TRAIN_ONLY: bool = True

_REQUIRE_EMBEDDING_DIM_MATCH: bool = True
_REQUIRED_EMBEDDING_DIM: int = 768
_REQUIRE_TIME_EVENT_ALIGNMENT: bool = True

_DEFAULT_EPS: float = 1.0e-12


class SurvivalError(RuntimeError):
    """Base exception for survival evaluation failures."""


class SurvivalSchemaError(SurvivalError):
    """Raised when survival input schema/shape contracts are violated."""


@dataclass(frozen=True)
class _FoldSpec:
    """Single train/test fold definition."""

    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


class SurvivalEvaluator:
    """Linear Cox PH evaluator for fold-based survival analysis."""

    def __init__(
        self,
        model_name: str = _MODEL_NAME,
        package_name: str = _MODEL_PACKAGE,
        center_embeddings: bool = _CENTER_EMBEDDINGS,
        l2_normalize_embeddings: bool = _L2_NORMALIZE_EMBEDDINGS,
        fit_statistics_on_train_only: bool = _FIT_STATS_ON_TRAIN_ONLY,
        embedding_service: Optional[Any] = None,
    ) -> None:
        model_name_value: str = str(model_name).strip()
        package_name_value: str = str(package_name).strip()
        if model_name_value != _MODEL_NAME:
            raise ValueError(f"model_name must be '{_MODEL_NAME}', got '{model_name_value}'.")
        if package_name_value != _MODEL_PACKAGE:
            raise ValueError(
                f"package_name must be '{_MODEL_PACKAGE}', got '{package_name_value}'."
            )

        if bool(center_embeddings) != _CENTER_EMBEDDINGS:
            raise ValueError(
                f"center_embeddings must be {_CENTER_EMBEDDINGS}, got {center_embeddings}."
            )
        if bool(l2_normalize_embeddings) != _L2_NORMALIZE_EMBEDDINGS:
            raise ValueError(
                "l2_normalize_embeddings must be "
                f"{_L2_NORMALIZE_EMBEDDINGS}, got {l2_normalize_embeddings}."
            )
        if bool(fit_statistics_on_train_only) != _FIT_STATS_ON_TRAIN_ONLY:
            raise ValueError(
                "fit_statistics_on_train_only must be "
                f"{_FIT_STATS_ON_TRAIN_ONLY}, got {fit_statistics_on_train_only}."
            )

        if embedding_service is not None:
            center_fn: Any = getattr(embedding_service, "center_and_l2", None)
            if not callable(center_fn):
                raise TypeError(
                    "embedding_service must implement callable center_and_l2(x, mean_vec=None)."
                )

        self.model_name: str = model_name_value
        self.package_name: str = package_name_value
        self.center_embeddings: bool = bool(center_embeddings)
        self.l2_normalize_embeddings: bool = bool(l2_normalize_embeddings)
        self.fit_statistics_on_train_only: bool = bool(fit_statistics_on_train_only)
        self.embedding_service: Optional[Any] = embedding_service

        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

    def run_survival(
        self,
        features: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        folds: list,
    ) -> dict:
        """Run fold-based survival evaluation with linear Cox PH."""
        x_all: np.ndarray = self._validate_features(features)
        time_all: np.ndarray = self._validate_time(time, expected_n=int(x_all.shape[0]))
        event_all: np.ndarray = self._validate_event(event, expected_n=int(x_all.shape[0]))
        if _REQUIRE_TIME_EVENT_ALIGNMENT and int(time_all.shape[0]) != int(event_all.shape[0]):
            raise SurvivalSchemaError(
                "time and event must have identical length, got "
                f"{time_all.shape[0]} and {event_all.shape[0]}."
            )

        fold_specs: List[_FoldSpec] = self._parse_folds(folds=folds, n_samples=int(x_all.shape[0]))
        if len(fold_specs) == 0:
            raise SurvivalSchemaError("At least one fold must be provided.")

        fold_ids: List[str] = []
        c_index_per_fold: List[float] = []
        fold_records: List[Dict[str, Any]] = []

        for fold_spec in fold_specs:
            train_idx: np.ndarray = fold_spec.train_idx
            test_idx: np.ndarray = fold_spec.test_idx

            x_train: np.ndarray = x_all[train_idx]
            x_test: np.ndarray = x_all[test_idx]
            time_train: np.ndarray = time_all[train_idx]
            time_test: np.ndarray = time_all[test_idx]
            event_train: np.ndarray = event_all[train_idx]
            event_test: np.ndarray = event_all[test_idx]

            if int(np.sum(event_train.astype(np.int64))) <= 0:
                raise SurvivalSchemaError(
                    f"{fold_spec.fold_id}: train split contains zero events; Cox fitting is undefined."
                )

            x_train_proc, x_test_proc = self._preprocess_fold_features(
                x_train=x_train,
                x_test=x_test,
            )
            y_train_struct: np.ndarray = self._to_survival_struct(
                time=time_train,
                event=event_train,
            )

            model: CoxPHSurvivalAnalysis = CoxPHSurvivalAnalysis()
            model.fit(x_train_proc, y_train_struct)

            risk_test: np.ndarray = np.asarray(model.predict(x_test_proc), dtype=np.float64)
            if risk_test.ndim != 1 or int(risk_test.shape[0]) != int(test_idx.shape[0]):
                raise SurvivalSchemaError(
                    f"{fold_spec.fold_id}: invalid risk output shape {tuple(risk_test.shape)}."
                )
            if not np.isfinite(risk_test).all():
                raise SurvivalSchemaError(
                    f"{fold_spec.fold_id}: non-finite risk values detected."
                )

            c_index_value: float = self._compute_c_index(
                time=time_test,
                event=event_test,
                risk=risk_test,
                fold_id=fold_spec.fold_id,
            )

            fold_ids.append(str(fold_spec.fold_id))
            c_index_per_fold.append(float(c_index_value))
            fold_records.append(
                {
                    "fold_id": str(fold_spec.fold_id),
                    "train_size": int(train_idx.shape[0]),
                    "test_size": int(test_idx.shape[0]),
                    "train_events": int(np.sum(event_train.astype(np.int64))),
                    "test_events": int(np.sum(event_test.astype(np.int64))),
                    "c_index": float(c_index_value),
                }
            )

        c_index_array: np.ndarray = np.asarray(c_index_per_fold, dtype=np.float64)
        c_index_mean: float = float(np.mean(c_index_array))
        c_index_std: float = (
            float(np.std(c_index_array, ddof=1)) if int(c_index_array.shape[0]) > 1 else 0.0
        )

        output: Dict[str, Any] = {
            "task_name": _TASK_NAME,
            "endpoint_name": _ENDPOINT_NAME,
            "num_samples": int(x_all.shape[0]),
            "num_events": int(np.sum(event_all.astype(np.int64))),
            "num_censored": int(x_all.shape[0]) - int(np.sum(event_all.astype(np.int64))),
            "fold_ids": fold_ids,
            "c_index_per_fold": [float(v) for v in c_index_per_fold],
            "c_index_mean": float(c_index_mean),
            "c_index_std": float(c_index_std),
            "preprocess_applied": {
                "center_embeddings": bool(self.center_embeddings),
                "l2_normalize_embeddings": bool(self.l2_normalize_embeddings),
                "fit_statistics_on_train_only": bool(self.fit_statistics_on_train_only),
            },
            "protocol": {
                "model": str(self.model_name),
                "package": str(self.package_name),
                "regularization_search_enabled": False,
            },
            "folds": fold_records,
        }
        return output

    def _preprocess_fold_features(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        train_mat: np.ndarray = np.asarray(x_train, dtype=np.float64)
        test_mat: np.ndarray = np.asarray(x_test, dtype=np.float64)

        if not self.center_embeddings and not self.l2_normalize_embeddings:
            return train_mat, test_mat

        if self.center_embeddings and self.l2_normalize_embeddings and self.embedding_service is not None:
            train_proc, mean_vec = self.embedding_service.center_and_l2(x=train_mat, mean_vec=None)
            test_proc, _ = self.embedding_service.center_and_l2(x=test_mat, mean_vec=mean_vec)
            return (
                np.asarray(train_proc, dtype=np.float64),
                np.asarray(test_proc, dtype=np.float64),
            )

        if self.center_embeddings:
            mean_vec_local: np.ndarray = np.mean(train_mat, axis=0, dtype=np.float64).astype(
                np.float64, copy=False
            )
            train_mat = train_mat - mean_vec_local.reshape(1, -1)
            test_mat = test_mat - mean_vec_local.reshape(1, -1)

        if self.l2_normalize_embeddings:
            train_norm: np.ndarray = np.linalg.norm(train_mat, ord=2, axis=1, keepdims=True)
            test_norm: np.ndarray = np.linalg.norm(test_mat, ord=2, axis=1, keepdims=True)
            train_norm = np.maximum(train_norm, float(_DEFAULT_EPS))
            test_norm = np.maximum(test_norm, float(_DEFAULT_EPS))
            train_mat = train_mat / train_norm
            test_mat = test_mat / test_norm

        return train_mat, test_mat

    @staticmethod
    def _to_survival_struct(time: np.ndarray, event: np.ndarray) -> np.ndarray:
        y_struct: np.ndarray = np.zeros(
            shape=(int(time.shape[0]),),
            dtype=[("event", np.bool_), ("time", np.float64)],
        )
        y_struct["event"] = event.astype(np.bool_, copy=False)
        y_struct["time"] = time.astype(np.float64, copy=False)
        return y_struct

    @staticmethod
    def _compute_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray, fold_id: str) -> float:
        try:
            c_index_raw: Tuple[float, int, int, int, int] = concordance_index_censored(
                event_indicator=event.astype(np.bool_, copy=False),
                event_time=time.astype(np.float64, copy=False),
                estimate=risk.astype(np.float64, copy=False),
            )
        except Exception as exc:  # noqa: BLE001
            raise SurvivalSchemaError(f"{fold_id}: failed to compute c-index: {exc}") from exc

        c_index_value: float = float(c_index_raw[0])
        if not np.isfinite(c_index_value):
            raise SurvivalSchemaError(f"{fold_id}: c-index is NaN/Inf.")
        return c_index_value

    def _parse_folds(self, folds: Any, n_samples: int) -> List[_FoldSpec]:
        if not isinstance(folds, list):
            raise TypeError(f"folds must be list, got {type(folds).__name__}.")
        if len(folds) == 0:
            raise SurvivalSchemaError("folds cannot be empty.")

        fold_specs: List[_FoldSpec] = []
        for idx, fold_obj in enumerate(folds):
            fold_id: str = f"fold_{idx}"
            train_spec: Any
            test_spec: Any

            if isinstance(fold_obj, Mapping):
                fold_id = str(fold_obj.get("fold_id", fold_obj.get("id", fold_id)))
                if "train" not in fold_obj or "test" not in fold_obj:
                    raise SurvivalSchemaError(
                        f"{fold_id}: each fold mapping must contain 'train' and 'test'."
                    )
                train_spec = fold_obj["train"]
                test_spec = fold_obj["test"]
            elif isinstance(fold_obj, (tuple, list)) and len(fold_obj) == 2:
                train_spec, test_spec = fold_obj[0], fold_obj[1]
            else:
                raise SurvivalSchemaError(
                    f"{fold_id}: unsupported fold spec type {type(fold_obj).__name__}."
                )

            train_idx: np.ndarray = self._to_indices(train_spec, n_samples=n_samples, name=f"{fold_id}.train")
            test_idx: np.ndarray = self._to_indices(test_spec, n_samples=n_samples, name=f"{fold_id}.test")

            if int(train_idx.shape[0]) <= 0:
                raise SurvivalSchemaError(f"{fold_id}: train split is empty.")
            if int(test_idx.shape[0]) <= 0:
                raise SurvivalSchemaError(f"{fold_id}: test split is empty.")

            overlap: set[int] = set(int(v) for v in train_idx.tolist()).intersection(
                int(v) for v in test_idx.tolist()
            )
            if len(overlap) > 0:
                raise SurvivalSchemaError(f"{fold_id}: train/test splits overlap.")

            fold_specs.append(
                _FoldSpec(
                    fold_id=str(fold_id),
                    train_idx=train_idx,
                    test_idx=test_idx,
                )
            )

        return fold_specs

    @staticmethod
    def _to_indices(spec: Any, n_samples: int, name: str) -> np.ndarray:
        if isinstance(spec, np.ndarray):
            arr: np.ndarray = spec
        elif isinstance(spec, (list, tuple)):
            arr = np.asarray(spec)
        else:
            raise SurvivalSchemaError(
                f"{name}: index spec must be np.ndarray/list/tuple, got {type(spec).__name__}."
            )

        if arr.ndim != 1:
            raise SurvivalSchemaError(f"{name}: index spec must be rank-1, got {tuple(arr.shape)}.")

        if arr.dtype == np.bool_:
            if int(arr.shape[0]) != int(n_samples):
                raise SurvivalSchemaError(
                    f"{name}: boolean mask length must be {n_samples}, got {int(arr.shape[0])}."
                )
            idx: np.ndarray = np.nonzero(arr)[0].astype(np.int64, copy=False)
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                if np.issubdtype(arr.dtype, np.floating):
                    if not np.all(np.equal(arr, np.floor(arr))):
                        raise SurvivalSchemaError(
                            f"{name}: floating indices must be integer-valued."
                        )
                    arr = arr.astype(np.int64)
                else:
                    raise SurvivalSchemaError(
                        f"{name}: indices must be integer or bool mask, got dtype={arr.dtype}."
                    )
            idx = np.asarray(arr, dtype=np.int64)

        if int(idx.shape[0]) == 0:
            return idx
        if np.any(idx < 0) or np.any(idx >= int(n_samples)):
            raise SurvivalSchemaError(
                f"{name}: indices out of range [0,{n_samples - 1}]."
            )

        unique_idx: np.ndarray = np.unique(idx)
        if int(unique_idx.shape[0]) != int(idx.shape[0]):
            raise SurvivalSchemaError(f"{name}: duplicate indices are not allowed.")
        return unique_idx

    @staticmethod
    def _validate_features(features: Any) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        x: np.ndarray = np.asarray(features, dtype=np.float64)

        if x.ndim != 2:
            raise SurvivalSchemaError(f"features must be rank-2 [N,D], got {tuple(x.shape)}.")
        if int(x.shape[0]) <= 1:
            raise SurvivalSchemaError("features must contain at least 2 samples.")
        if _REQUIRE_EMBEDDING_DIM_MATCH and int(x.shape[1]) != int(_REQUIRED_EMBEDDING_DIM):
            raise SurvivalSchemaError(
                "features second dimension mismatch: expected "
                f"{_REQUIRED_EMBEDDING_DIM}, got {int(x.shape[1])}."
            )
        if not np.isfinite(x).all():
            raise SurvivalSchemaError("features contain NaN/Inf values.")
        return x

    @staticmethod
    def _validate_time(time: Any, expected_n: int) -> np.ndarray:
        if not isinstance(time, np.ndarray):
            time = np.asarray(time)
        t: np.ndarray = np.asarray(time, dtype=np.float64)

        if t.ndim != 1:
            raise SurvivalSchemaError(f"time must be rank-1 [N], got {tuple(t.shape)}.")
        if int(t.shape[0]) != int(expected_n):
            raise SurvivalSchemaError(
                f"time length must match features rows ({expected_n}), got {int(t.shape[0])}."
            )
        if not np.isfinite(t).all():
            raise SurvivalSchemaError("time contains NaN/Inf values.")
        if np.any(t <= 0.0):
            raise SurvivalSchemaError("time values must be strictly positive for Cox PH.")
        return t

    @staticmethod
    def _validate_event(event: Any, expected_n: int) -> np.ndarray:
        if not isinstance(event, np.ndarray):
            event = np.asarray(event)
        e_raw: np.ndarray = np.asarray(event)

        if e_raw.ndim != 1:
            raise SurvivalSchemaError(f"event must be rank-1 [N], got {tuple(e_raw.shape)}.")
        if int(e_raw.shape[0]) != int(expected_n):
            raise SurvivalSchemaError(
                f"event length must match features rows ({expected_n}), got {int(e_raw.shape[0])}."
            )

        if np.issubdtype(e_raw.dtype, np.bool_):
            e_int: np.ndarray = e_raw.astype(np.int64, copy=False)
        elif np.issubdtype(e_raw.dtype, np.integer):
            e_int = e_raw.astype(np.int64, copy=False)
        elif np.issubdtype(e_raw.dtype, np.floating):
            if not np.all(np.equal(e_raw, np.floor(e_raw))):
                raise SurvivalSchemaError("event floating values must be integer-valued.")
            e_int = e_raw.astype(np.int64, copy=False)
        else:
            raise SurvivalSchemaError(
                f"event dtype must be bool/int/float, got {e_raw.dtype}."
            )

        unique_values: np.ndarray = np.unique(e_int)
        valid_values: set[int] = {0, 1}
        if not set(int(v) for v in unique_values.tolist()).issubset(valid_values):
            raise SurvivalSchemaError(
                f"event values must be binary in {{0,1}}, got {unique_values.tolist()}."
            )

        if int(np.sum(e_int)) <= 0:
            raise SurvivalSchemaError("event contains zero positive events in the full set.")

        return e_int.astype(np.int64, copy=False)


class Evaluator(SurvivalEvaluator):
    """Design-compat alias exposing run_survival."""


def run_survival(features: np.ndarray, time: np.ndarray, event: np.ndarray, folds: list) -> dict:
    """Convenience functional API for survival evaluation."""
    evaluator: SurvivalEvaluator = SurvivalEvaluator()
    return evaluator.run_survival(features=features, time=time, event=event, folds=folds)


__all__ = [
    "SurvivalError",
    "SurvivalSchemaError",
    "SurvivalEvaluator",
    "Evaluator",
    "run_survival",
]

