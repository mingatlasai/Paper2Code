"""Survival linear-probe evaluator for THREADS downstream tasks.

This module implements the design-locked public interface:
- ``SurvivalEvaluator.__init__(alpha: float, max_iter: int) -> None``
- ``SurvivalEvaluator.fit(x_train: object, y_time: object, y_event: object) -> None``
- ``SurvivalEvaluator.predict_risk(x_test: object) -> object``
- ``SurvivalEvaluator.score_c_index(y_time: object, y_event: object, risk: object) -> float``

Paper/config alignment:
- Estimator: ``sksurv.linear_model.CoxnetSurvivalAnalysis``
- Fixed default ``max_iter=10000``
- Alpha defaults:
  - overall survival (OS): ``0.07``
  - progression-free survival (PFS): ``0.01``
- Task/model alpha overrides:
  - ("CPTAC-CCRCC overall survival", "CHIEF") -> ``0.01``
  - ("BOEHMK progression-free survival", "PRISM") -> ``0.02``
- Metric: concordance index (c-index), delegated to ``src.utils.metrics.score_c_index``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sksurv.linear_model import CoxnetSurvivalAnalysis

from src.utils.metrics import MetricContext, score_c_index as metric_score_c_index


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_ALPHA_OS: float = 0.07
DEFAULT_ALPHA_PFS: float = 0.01
DEFAULT_MAX_ITER: int = 10000

DEFAULT_METRIC_NAME_C_INDEX: str = "c_index"
DEFAULT_VALIDATE_NUMERICS: bool = True

DEFAULT_TOLERANCE: float = 1.0e-7
DEFAULT_NORMALIZE: bool = False
DEFAULT_COPY_X: bool = True
DEFAULT_VERBOSE: bool = False

DEFAULT_SURVIVAL_TIME_FIELD: str = "time"
DEFAULT_SURVIVAL_EVENT_FIELD: str = "event"

_ENDPOINT_OVERALL_SURVIVAL: str = "overall_survival"
_ENDPOINT_PROGRESSION_FREE_SURVIVAL: str = "progression_free_survival"

PFS_KEYWORDS: Tuple[str, ...] = (
    "progression-free survival",
    "progression free survival",
    "pfs",
)
OS_KEYWORDS: Tuple[str, ...] = (
    "overall survival",
    "os",
)

ALPHA_OVERRIDES: Tuple[Tuple[str, str, float], ...] = (
    ("CPTAC-CCRCC overall survival", "CHIEF", 0.01),
    ("BOEHMK progression-free survival", "PRISM", 0.02),
)


class SurvivalEvaluatorError(Exception):
    """Base exception for survival evaluator failures."""


class SurvivalEvaluatorConfigError(SurvivalEvaluatorError):
    """Raised when evaluator configuration is invalid."""


class SurvivalEvaluatorInputError(SurvivalEvaluatorError):
    """Raised when fit/predict/score inputs are malformed."""


class SurvivalEvaluatorNotFittedError(SurvivalEvaluatorError):
    """Raised when prediction/scoring requires a fitted estimator."""


@dataclass(frozen=True)
class _FitState:
    """Internal estimator fit state metadata."""

    is_fitted: bool
    n_samples: int
    n_features: int
    alpha: float
    max_iter: int
    convergence_warning_seen: bool


class SurvivalEvaluator:
    """CoxNet evaluator for survival linear probing."""

    def __init__(self, alpha: float = DEFAULT_ALPHA_OS, max_iter: int = DEFAULT_MAX_ITER) -> None:
        """Initialize survival evaluator.

        Args:
            alpha: CoxNet alpha regularization value.
            max_iter: Maximum solver iterations.
        """
        self._alpha: float = self._validate_alpha(alpha)
        self._max_iter: int = self._validate_max_iter(max_iter)

        self._estimator: Optional[CoxnetSurvivalAnalysis] = None
        self._fit_state: _FitState = _FitState(
            is_fitted=False,
            n_samples=0,
            n_features=0,
            alpha=self._alpha,
            max_iter=self._max_iter,
            convergence_warning_seen=False,
        )

        self._validate_numerics: bool = DEFAULT_VALIDATE_NUMERICS

    def fit(self, x_train: object, y_time: object, y_event: object) -> None:
        """Fit CoxNet on one fold.

        Args:
            x_train: Feature matrix-like input ``[N, F]``.
            y_time: Survival times (positive).
            y_event: Event indicator (1=event, 0=censored).
        """
        x_train_array: np.ndarray = self._coerce_features(x_train, name="x_train")
        time_array: np.ndarray = self._coerce_time(y_time, name="y_time")
        event_array: np.ndarray = self._coerce_event(y_event, name="y_event")

        if int(x_train_array.shape[0]) != int(time_array.shape[0]):
            raise SurvivalEvaluatorInputError(
                "x_train and y_time sample counts mismatch: "
                f"{int(x_train_array.shape[0])} vs {int(time_array.shape[0])}."
            )
        if int(x_train_array.shape[0]) != int(event_array.shape[0]):
            raise SurvivalEvaluatorInputError(
                "x_train and y_event sample counts mismatch: "
                f"{int(x_train_array.shape[0])} vs {int(event_array.shape[0])}."
            )

        if int(x_train_array.shape[0]) < 2:
            raise SurvivalEvaluatorInputError("At least two samples are required for survival fitting.")
        if int(np.sum(event_array.astype(np.int64))) <= 0:
            raise SurvivalEvaluatorInputError(
                "At least one event case is required for CoxNet fitting (all samples censored)."
            )

        y_structured: np.ndarray = self._build_structured_survival_array(
            time_array=time_array,
            event_array=event_array,
        )

        estimator: CoxnetSurvivalAnalysis = CoxnetSurvivalAnalysis(
            alphas=np.asarray([float(self._alpha)], dtype=np.float64),
            n_alphas=1,
            alpha_min_ratio=1.0,
            max_iter=int(self._max_iter),
            tol=float(DEFAULT_TOLERANCE),
            normalize=bool(DEFAULT_NORMALIZE),
            copy_X=bool(DEFAULT_COPY_X),
            verbose=bool(DEFAULT_VERBOSE),
            fit_baseline_model=False,
        )

        convergence_warning_seen: bool = False
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=ConvergenceWarning)
            estimator.fit(x_train_array, y_structured)
            convergence_warning_seen = any(
                issubclass(item.category, ConvergenceWarning) for item in caught_warnings
            )

        self._estimator = estimator
        self._fit_state = _FitState(
            is_fitted=True,
            n_samples=int(x_train_array.shape[0]),
            n_features=int(x_train_array.shape[1]),
            alpha=float(self._alpha),
            max_iter=int(self._max_iter),
            convergence_warning_seen=bool(convergence_warning_seen),
        )

    def predict_risk(self, x_test: object) -> object:
        """Predict continuous risk scores for test samples.

        Args:
            x_test: Feature matrix-like input ``[N, F]``.

        Returns:
            ``np.ndarray`` with shape ``[N]``.
        """
        estimator: CoxnetSurvivalAnalysis = self._require_fitted_estimator()
        x_test_array: np.ndarray = self._coerce_features(x_test, name="x_test")

        if int(x_test_array.shape[1]) != int(self._fit_state.n_features):
            raise SurvivalEvaluatorInputError(
                "x_test feature width mismatch: "
                f"expected {self._fit_state.n_features}, got {int(x_test_array.shape[1])}."
            )

        try:
            risk_array: np.ndarray = np.asarray(
                estimator.predict(x_test_array, alpha=float(self._alpha)),
                dtype=np.float64,
            )
        except TypeError:
            # Compatibility fallback for versions where predict() does not expose alpha argument.
            risk_array = np.asarray(estimator.predict(x_test_array), dtype=np.float64)

        if risk_array.ndim == 0:
            risk_array = risk_array.reshape(1)
        if risk_array.ndim > 1:
            risk_array = risk_array.reshape(-1)

        if int(risk_array.shape[0]) != int(x_test_array.shape[0]):
            raise SurvivalEvaluatorInputError(
                "predict_risk output length mismatch: "
                f"expected {int(x_test_array.shape[0])}, got {int(risk_array.shape[0])}."
            )

        if self._validate_numerics and not np.isfinite(risk_array).all():
            raise SurvivalEvaluatorInputError("predict_risk returned NaN/Inf values.")

        return risk_array.astype(np.float64, copy=False)

    def score_c_index(self, y_time: object, y_event: object, risk: object) -> float:
        """Compute concordance index (c-index).

        Args:
            y_time: Survival times.
            y_event: Event indicators.
            risk: Predicted risk scores.

        Returns:
            c-index as float.
        """
        time_array: np.ndarray = self._coerce_time(y_time, name="y_time")
        event_array: np.ndarray = self._coerce_event(y_event, name="y_event")
        risk_array: np.ndarray = self._coerce_risk(risk, name="risk")

        if int(time_array.shape[0]) != int(event_array.shape[0]):
            raise SurvivalEvaluatorInputError(
                "y_time and y_event sample count mismatch: "
                f"{int(time_array.shape[0])} vs {int(event_array.shape[0])}."
            )
        if int(time_array.shape[0]) != int(risk_array.shape[0]):
            raise SurvivalEvaluatorInputError(
                "y_time and risk sample count mismatch: "
                f"{int(time_array.shape[0])} vs {int(risk_array.shape[0])}."
            )

        context: MetricContext = MetricContext(task_name="survival")
        c_index_value: float = float(
            metric_score_c_index(
                y_time=time_array,
                y_event=event_array,
                risk_score=risk_array,
                higher_score_higher_risk=True,
                context=context,
            )
        )

        if not np.isfinite(c_index_value):
            raise SurvivalEvaluatorInputError("c-index scoring produced non-finite value.")

        return c_index_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_fitted_estimator(self) -> CoxnetSurvivalAnalysis:
        if self._estimator is None or not self._fit_state.is_fitted:
            raise SurvivalEvaluatorNotFittedError("Estimator is not fitted. Call fit() first.")
        return self._estimator

    def _coerce_features(self, value: object, name: str) -> np.ndarray:
        """Coerce feature payload to dense float64 matrix [N, F]."""
        array_value: np.ndarray

        if isinstance(value, np.ndarray):
            array_value = value
        elif hasattr(value, "to_numpy") and callable(getattr(value, "to_numpy")):
            array_value = np.asarray(value.to_numpy())
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise SurvivalEvaluatorInputError(f"{name} cannot be empty.")
            first_item: Any = value[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                array_value = np.asarray([np.asarray(item) for item in value], dtype=np.float64)
            else:
                array_value = np.asarray(value)
        else:
            try:
                array_value = np.asarray(value)
            except Exception as exc:  # noqa: BLE001
                raise SurvivalEvaluatorInputError(
                    f"{name} cannot be converted to numpy array: {exc}"
                ) from exc

        if array_value.dtype == object and array_value.ndim == 1:
            if array_value.size == 0:
                raise SurvivalEvaluatorInputError(f"{name} cannot be empty.")
            try:
                stacked: np.ndarray = np.vstack(
                    [np.asarray(item, dtype=np.float64).reshape(-1) for item in array_value.tolist()]
                )
            except Exception as exc:  # noqa: BLE001
                raise SurvivalEvaluatorInputError(
                    f"{name} object payload cannot be stacked into 2D feature matrix: {exc}"
                ) from exc
            array_value = stacked

        if array_value.ndim == 1:
            array_value = array_value.reshape(-1, 1)

        if array_value.ndim != 2:
            raise SurvivalEvaluatorInputError(
                f"{name} must be rank-2 feature matrix [N,F], got shape={tuple(array_value.shape)}."
            )

        if int(array_value.shape[0]) <= 0:
            raise SurvivalEvaluatorInputError(f"{name} must contain at least one sample.")
        if int(array_value.shape[1]) <= 0:
            raise SurvivalEvaluatorInputError(f"{name} must contain at least one feature.")

        feature_matrix: np.ndarray = np.asarray(array_value, dtype=np.float64)
        if self._validate_numerics and not np.isfinite(feature_matrix).all():
            raise SurvivalEvaluatorInputError(f"{name} contains NaN/Inf values.")

        return feature_matrix

    def _coerce_time(self, value: object, name: str) -> np.ndarray:
        """Coerce survival time to finite positive float64 vector [N]."""
        try:
            time_array: np.ndarray = np.asarray(value, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            raise SurvivalEvaluatorInputError(
                f"{name} cannot be converted to float array: {exc}"
            ) from exc

        if time_array.ndim == 0:
            time_array = time_array.reshape(1)
        if time_array.ndim > 1:
            time_array = time_array.reshape(-1)

        if int(time_array.shape[0]) <= 0:
            raise SurvivalEvaluatorInputError(f"{name} cannot be empty.")
        if self._validate_numerics and not np.isfinite(time_array).all():
            raise SurvivalEvaluatorInputError(f"{name} contains NaN/Inf values.")
        if np.any(time_array <= 0.0):
            raise SurvivalEvaluatorInputError(f"{name} must contain strictly positive values.")

        return time_array.astype(np.float64, copy=False)

    def _coerce_event(self, value: object, name: str) -> np.ndarray:
        """Coerce event indicator to boolean vector [N]."""
        try:
            event_array: np.ndarray = np.asarray(value)
        except Exception as exc:  # noqa: BLE001
            raise SurvivalEvaluatorInputError(
                f"{name} cannot be converted to array: {exc}"
            ) from exc

        if event_array.ndim == 0:
            event_array = event_array.reshape(1)
        if event_array.ndim > 1:
            event_array = event_array.reshape(-1)

        if int(event_array.shape[0]) <= 0:
            raise SurvivalEvaluatorInputError(f"{name} cannot be empty.")

        if event_array.dtype == np.bool_:
            return event_array.astype(bool, copy=False)

        if np.issubdtype(event_array.dtype, np.number):
            numeric: np.ndarray = np.asarray(event_array, dtype=np.float64)
            if self._validate_numerics and not np.isfinite(numeric).all():
                raise SurvivalEvaluatorInputError(f"{name} contains NaN/Inf values.")
            unique_values: set[float] = set(np.unique(numeric).tolist())
            if not unique_values.issubset({0.0, 1.0}):
                raise SurvivalEvaluatorInputError(
                    f"{name} must contain only 0/1 for numeric input, got {sorted(unique_values)}."
                )
            return numeric.astype(bool, copy=False)

        normalized_values: List[bool] = []
        for item in event_array.tolist():
            token: str = str(item).strip().lower()
            if token in {"1", "true", "t", "yes", "y", "event"}:
                normalized_values.append(True)
            elif token in {"0", "false", "f", "no", "n", "censored"}:
                normalized_values.append(False)
            else:
                raise SurvivalEvaluatorInputError(
                    f"{name} has unsupported categorical token: {item!r}."
                )

        return np.asarray(normalized_values, dtype=bool)

    def _coerce_risk(self, value: object, name: str) -> np.ndarray:
        """Coerce risk payload to finite float64 vector [N]."""
        try:
            risk_array: np.ndarray = np.asarray(value, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            raise SurvivalEvaluatorInputError(
                f"{name} cannot be converted to float array: {exc}"
            ) from exc

        if risk_array.ndim == 0:
            risk_array = risk_array.reshape(1)
        if risk_array.ndim > 1:
            risk_array = risk_array.reshape(-1)

        if int(risk_array.shape[0]) <= 0:
            raise SurvivalEvaluatorInputError(f"{name} cannot be empty.")
        if self._validate_numerics and not np.isfinite(risk_array).all():
            raise SurvivalEvaluatorInputError(f"{name} contains NaN/Inf values.")

        return risk_array.astype(np.float64, copy=False)

    def _build_structured_survival_array(
        self,
        *,
        time_array: np.ndarray,
        event_array: np.ndarray,
    ) -> np.ndarray:
        """Build sksurv-compatible structured target array."""
        dtype: np.dtype = np.dtype(
            [
                (DEFAULT_SURVIVAL_EVENT_FIELD, np.bool_),
                (DEFAULT_SURVIVAL_TIME_FIELD, np.float64),
            ]
        )
        structured: np.ndarray = np.empty(int(time_array.shape[0]), dtype=dtype)
        structured[DEFAULT_SURVIVAL_EVENT_FIELD] = event_array.astype(bool, copy=False)
        structured[DEFAULT_SURVIVAL_TIME_FIELD] = time_array.astype(np.float64, copy=False)
        return structured

    @staticmethod
    def _validate_alpha(value: float) -> float:
        if isinstance(value, bool):
            raise SurvivalEvaluatorConfigError("alpha must be float, got bool.")
        try:
            alpha_value: float = float(value)
        except Exception as exc:  # noqa: BLE001
            raise SurvivalEvaluatorConfigError(f"alpha must be float, got {value!r}.") from exc

        if not np.isfinite(alpha_value) or alpha_value <= 0.0:
            raise SurvivalEvaluatorConfigError(
                f"alpha must be finite and > 0, got {alpha_value}."
            )
        return alpha_value

    @staticmethod
    def _validate_max_iter(value: int) -> int:
        if isinstance(value, bool):
            raise SurvivalEvaluatorConfigError("max_iter must be int, got bool.")
        try:
            max_iter_value: int = int(value)
        except Exception as exc:  # noqa: BLE001
            raise SurvivalEvaluatorConfigError(f"max_iter must be int, got {value!r}.") from exc
        if max_iter_value <= 0:
            raise SurvivalEvaluatorConfigError(
                f"max_iter must be > 0, got {max_iter_value}."
            )
        return max_iter_value


# -----------------------------------------------------------------------------
# Alpha policy helpers (for upstream evaluators/pipelines)
# -----------------------------------------------------------------------------
def resolve_endpoint_type(task_name: str) -> str:
    """Resolve endpoint type from task name using paper-config keywords."""
    normalized_task_name: str = "" if task_name is None else str(task_name).strip().lower()

    for keyword in PFS_KEYWORDS:
        if keyword in normalized_task_name:
            return _ENDPOINT_PROGRESSION_FREE_SURVIVAL
    for keyword in OS_KEYWORDS:
        if keyword in normalized_task_name:
            return _ENDPOINT_OVERALL_SURVIVAL

    return _ENDPOINT_OVERALL_SURVIVAL


def resolve_alpha(
    *,
    task_name: str,
    model_name: str,
    alpha_overall_survival: float = DEFAULT_ALPHA_OS,
    alpha_progression_free_survival: float = DEFAULT_ALPHA_PFS,
    alpha_overrides: Sequence[Tuple[str, str, float]] = ALPHA_OVERRIDES,
) -> float:
    """Resolve CoxNet alpha from fixed defaults and task/model overrides.

    Resolution order:
    1) Exact task+model override match.
    2) Endpoint keyword-based default (PFS or OS).
    3) OS fallback.
    """
    normalized_task_name: str = "" if task_name is None else str(task_name).strip().lower()
    normalized_model_name: str = "" if model_name is None else str(model_name).strip().lower()

    for override_task, override_model, override_alpha in alpha_overrides:
        if normalized_task_name == str(override_task).strip().lower() and normalized_model_name == str(override_model).strip().lower():
            return SurvivalEvaluator._validate_alpha(float(override_alpha))

    endpoint_type: str = resolve_endpoint_type(task_name=task_name)
    if endpoint_type == _ENDPOINT_PROGRESSION_FREE_SURVIVAL:
        return SurvivalEvaluator._validate_alpha(float(alpha_progression_free_survival))

    return SurvivalEvaluator._validate_alpha(float(alpha_overall_survival))


__all__ = [
    "DEFAULT_ALPHA_OS",
    "DEFAULT_ALPHA_PFS",
    "DEFAULT_MAX_ITER",
    "DEFAULT_METRIC_NAME_C_INDEX",
    "ALPHA_OVERRIDES",
    "SurvivalEvaluatorError",
    "SurvivalEvaluatorConfigError",
    "SurvivalEvaluatorInputError",
    "SurvivalEvaluatorNotFittedError",
    "SurvivalEvaluator",
    "resolve_endpoint_type",
    "resolve_alpha",
]
