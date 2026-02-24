"""Linear probing evaluator for THREADS downstream classification tasks.

This module implements the design-locked public interface:
- ``LinearProbeEvaluator.__init__(c_value, max_iter, solver, class_weight)``
- ``LinearProbeEvaluator.fit(x_train, y_train)``
- ``LinearProbeEvaluator.predict_proba(x_test)``
- ``LinearProbeEvaluator.score_binary_auc(y_true, y_prob)``
- ``LinearProbeEvaluator.score_multiclass_bacc(y_true, y_pred)``
- ``LinearProbeEvaluator.score_qwk(y_true, y_pred)``

Paper/config alignment:
- LogisticRegression fixed defaults:
  - ``C=0.5``
  - ``solver='lbfgs'``
  - ``max_iter=10000``
  - ``class_weight='balanced'``
- No hyperparameter search/tuning in this evaluator.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.utils.metrics import (
    MetricContext,
    score_binary_auc as metric_score_binary_auc,
    score_multiclass_balanced_accuracy,
    score_qwk as metric_score_qwk,
)


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_C_VALUE: float = 0.5
DEFAULT_MAX_ITER: int = 10000
DEFAULT_SOLVER: str = "lbfgs"
DEFAULT_CLASS_WEIGHT: str = "balanced"
DEFAULT_RANDOM_STATE: int = 42

DEFAULT_METRIC_NAME_AUC: str = "macro_auc"
DEFAULT_METRIC_NAME_BACC: str = "balanced_accuracy"
DEFAULT_METRIC_NAME_QWK: str = "quadratic_weighted_kappa"

_ALLOWED_SOLVERS: Tuple[str, ...] = ("lbfgs", "newton-cg", "newton-cholesky", "sag", "saga", "liblinear")


class LinearProbeError(Exception):
    """Base exception for linear probe evaluator failures."""


class LinearProbeConfigError(LinearProbeError):
    """Raised when evaluator configuration is invalid."""


class LinearProbeInputError(LinearProbeError):
    """Raised when fit/predict/score inputs are malformed."""


class LinearProbeNotFittedError(LinearProbeError):
    """Raised when prediction/scoring requires a fitted estimator."""


@dataclass(frozen=True)
class _FitState:
    """Internal estimator fit state metadata."""

    is_fitted: bool
    n_samples: int
    n_features: int
    n_classes: int
    classes: Tuple[Any, ...]
    convergence_warning_seen: bool


class LinearProbeEvaluator:
    """Fixed-parameter logistic-regression evaluator for linear probing."""

    def __init__(
        self,
        c_value: float = DEFAULT_C_VALUE,
        max_iter: int = DEFAULT_MAX_ITER,
        solver: str = DEFAULT_SOLVER,
        class_weight: str = DEFAULT_CLASS_WEIGHT,
    ) -> None:
        """Initialize evaluator with explicit configuration.

        Args:
            c_value: Inverse regularization strength ``C``.
            max_iter: Maximum optimizer iterations.
            solver: LogisticRegression solver.
            class_weight: Class-weight policy.
        """
        self._c_value: float = self._validate_c_value(c_value)
        self._max_iter: int = self._validate_max_iter(max_iter)
        self._solver: str = self._validate_solver(solver)
        self._class_weight: str = self._validate_class_weight(class_weight)

        self._estimator: Optional[LogisticRegression] = None
        self._fit_state: _FitState = _FitState(
            is_fitted=False,
            n_samples=0,
            n_features=0,
            n_classes=0,
            classes=tuple(),
            convergence_warning_seen=False,
        )

    def fit(self, x_train: object, y_train: object) -> None:
        """Fit fixed-parameter logistic regression on one fold.

        Args:
            x_train: Training feature matrix-like input.
            y_train: Training labels.
        """
        x_train_array: np.ndarray = self._coerce_features(x_train, name="x_train")
        y_train_array: np.ndarray = self._coerce_labels(y_train, name="y_train")

        if int(x_train_array.shape[0]) != int(y_train_array.shape[0]):
            raise LinearProbeInputError(
                "x_train and y_train sample counts mismatch: "
                f"{int(x_train_array.shape[0])} vs {int(y_train_array.shape[0])}."
            )

        unique_classes: np.ndarray = np.unique(y_train_array)
        if int(unique_classes.shape[0]) < 2:
            raise LinearProbeInputError(
                "y_train must contain at least two classes for logistic regression; "
                f"got classes={unique_classes.tolist()}."
            )

        estimator: LogisticRegression = LogisticRegression(
            C=float(self._c_value),
            solver=str(self._solver),
            max_iter=int(self._max_iter),
            class_weight=str(self._class_weight),
            random_state=DEFAULT_RANDOM_STATE,
            multi_class="auto",
        )

        convergence_warning_seen: bool = False
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", category=ConvergenceWarning)
            estimator.fit(x_train_array, y_train_array)
            convergence_warning_seen = any(
                issubclass(item.category, ConvergenceWarning) for item in caught_warnings
            )

        classes_tuple: Tuple[Any, ...] = tuple(estimator.classes_.tolist())
        self._estimator = estimator
        self._fit_state = _FitState(
            is_fitted=True,
            n_samples=int(x_train_array.shape[0]),
            n_features=int(x_train_array.shape[1]),
            n_classes=int(len(classes_tuple)),
            classes=classes_tuple,
            convergence_warning_seen=bool(convergence_warning_seen),
        )

    def predict_proba(self, x_test: object) -> object:
        """Predict class probabilities for test samples.

        Args:
            x_test: Test feature matrix-like input.

        Returns:
            ``np.ndarray`` of shape ``[N, C]``.
        """
        estimator: LogisticRegression = self._require_fitted_estimator()
        x_test_array: np.ndarray = self._coerce_features(x_test, name="x_test")

        if int(x_test_array.shape[1]) != int(self._fit_state.n_features):
            raise LinearProbeInputError(
                "x_test feature width mismatch: "
                f"expected {self._fit_state.n_features}, got {int(x_test_array.shape[1])}."
            )

        probability: np.ndarray = estimator.predict_proba(x_test_array)
        if probability.ndim != 2:
            raise LinearProbeInputError(
                f"predict_proba returned invalid shape {tuple(probability.shape)}."
            )
        if int(probability.shape[0]) != int(x_test_array.shape[0]):
            raise LinearProbeInputError(
                "predict_proba row count mismatch: "
                f"expected {int(x_test_array.shape[0])}, got {int(probability.shape[0])}."
            )
        if int(probability.shape[1]) != int(self._fit_state.n_classes):
            raise LinearProbeInputError(
                "predict_proba class count mismatch: "
                f"expected {self._fit_state.n_classes}, got {int(probability.shape[1])}."
            )
        if not np.isfinite(probability).all():
            raise LinearProbeInputError("predict_proba returned NaN/Inf values.")

        return probability.astype(np.float64, copy=False)

    def score_binary_auc(self, y_true: object, y_prob: object) -> float:
        """Score binary classification using AUC.

        Args:
            y_true: Ground-truth labels.
            y_prob: Positive-class scores or probability matrix.

        Returns:
            AUC value as float.
        """
        y_true_array: np.ndarray = self._coerce_labels(y_true, name="y_true")
        y_prob_array: np.ndarray = self._coerce_score_array(y_prob, name="y_prob")

        if int(y_prob_array.shape[0]) != int(y_true_array.shape[0]):
            raise LinearProbeInputError(
                "y_true/y_prob sample mismatch for AUC scoring: "
                f"{int(y_true_array.shape[0])} vs {int(y_prob_array.shape[0])}."
            )

        context: MetricContext = MetricContext(task_name="binary_classification")
        auc_value: float = float(
            metric_score_binary_auc(
                y_true=y_true_array,
                y_score=y_prob_array,
                context=context,
            )
        )

        if not np.isfinite(auc_value):
            raise LinearProbeInputError("AUC scoring produced non-finite value.")
        return auc_value

    def score_multiclass_bacc(self, y_true: object, y_pred: object) -> float:
        """Score multiclass subtyping with balanced accuracy."""
        y_true_array: np.ndarray = self._coerce_labels(y_true, name="y_true")
        y_pred_array: np.ndarray = self._coerce_labels(y_pred, name="y_pred")

        if int(y_pred_array.shape[0]) != int(y_true_array.shape[0]):
            raise LinearProbeInputError(
                "y_true/y_pred sample mismatch for balanced-accuracy scoring: "
                f"{int(y_true_array.shape[0])} vs {int(y_pred_array.shape[0])}."
            )

        context: MetricContext = MetricContext(task_name="multiclass_subtyping")
        bacc_value: float = float(
            score_multiclass_balanced_accuracy(
                y_true=y_true_array,
                y_pred=y_pred_array,
                context=context,
            )
        )

        if not np.isfinite(bacc_value):
            raise LinearProbeInputError("Balanced-accuracy scoring produced non-finite value.")
        return bacc_value

    def score_qwk(self, y_true: object, y_pred: object) -> float:
        """Score grading predictions with quadratic weighted kappa."""
        y_true_array: np.ndarray = self._coerce_labels(y_true, name="y_true")
        y_pred_array: np.ndarray = self._coerce_labels(y_pred, name="y_pred")

        if int(y_pred_array.shape[0]) != int(y_true_array.shape[0]):
            raise LinearProbeInputError(
                "y_true/y_pred sample mismatch for QWK scoring: "
                f"{int(y_true_array.shape[0])} vs {int(y_pred_array.shape[0])}."
            )

        context: MetricContext = MetricContext(task_name="grading")
        qwk_value: float = float(
            metric_score_qwk(
                y_true=y_true_array,
                y_pred=y_pred_array,
                context=context,
            )
        )

        if not np.isfinite(qwk_value):
            raise LinearProbeInputError("QWK scoring produced non-finite value.")
        return qwk_value

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _require_fitted_estimator(self) -> LogisticRegression:
        if self._estimator is None or not self._fit_state.is_fitted:
            raise LinearProbeNotFittedError("Estimator is not fitted. Call fit() first.")
        return self._estimator

    def _coerce_features(self, value: object, name: str) -> np.ndarray:
        """Coerce feature payload to dense float64 matrix [N, F]."""
        array_value: np.ndarray

        if isinstance(value, np.ndarray):
            array_value = value
        elif hasattr(value, "to_numpy") and callable(getattr(value, "to_numpy")):
            # pandas DataFrame/Series and similar containers.
            array_value = np.asarray(value.to_numpy())
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                raise LinearProbeInputError(f"{name} cannot be empty.")
            first_item: Any = value[0]
            if isinstance(first_item, (list, tuple, np.ndarray)):
                array_value = np.asarray([np.asarray(item) for item in value], dtype=np.float64)
            else:
                array_value = np.asarray(value)
        else:
            try:
                array_value = np.asarray(value)
            except Exception as exc:  # noqa: BLE001
                raise LinearProbeInputError(
                    f"{name} cannot be converted to numpy array: {exc}"
                ) from exc

        # Common path: object array of embedding vectors.
        if array_value.dtype == object and array_value.ndim == 1:
            if array_value.size == 0:
                raise LinearProbeInputError(f"{name} cannot be empty.")
            try:
                stacked: np.ndarray = np.vstack(
                    [np.asarray(item, dtype=np.float64).reshape(-1) for item in array_value.tolist()]
                )
            except Exception as exc:  # noqa: BLE001
                raise LinearProbeInputError(
                    f"{name} object payload cannot be stacked into 2D feature matrix: {exc}"
                ) from exc
            array_value = stacked

        if array_value.ndim == 1:
            array_value = array_value.reshape(-1, 1)

        if array_value.ndim != 2:
            raise LinearProbeInputError(
                f"{name} must be rank-2 feature matrix [N,F], got shape={tuple(array_value.shape)}."
            )

        if int(array_value.shape[0]) <= 0:
            raise LinearProbeInputError(f"{name} must contain at least one sample.")
        if int(array_value.shape[1]) <= 0:
            raise LinearProbeInputError(f"{name} must contain at least one feature.")

        feature_matrix: np.ndarray = np.asarray(array_value, dtype=np.float64)
        if not np.isfinite(feature_matrix).all():
            raise LinearProbeInputError(f"{name} contains NaN/Inf values.")

        return feature_matrix

    def _coerce_labels(self, value: object, name: str) -> np.ndarray:
        """Coerce labels to 1D numpy array preserving label semantics."""
        try:
            label_array: np.ndarray = np.asarray(value)
        except Exception as exc:  # noqa: BLE001
            raise LinearProbeInputError(
                f"{name} cannot be converted to numpy array: {exc}"
            ) from exc

        if label_array.ndim == 0:
            label_array = label_array.reshape(1)

        if label_array.ndim > 1:
            label_array = label_array.reshape(-1)

        if int(label_array.shape[0]) <= 0:
            raise LinearProbeInputError(f"{name} cannot be empty.")

        # Ensure finite for numeric labels, keep strings/objects as-is.
        if np.issubdtype(label_array.dtype, np.number):
            numeric_labels: np.ndarray = np.asarray(label_array, dtype=np.float64)
            if not np.isfinite(numeric_labels).all():
                raise LinearProbeInputError(f"{name} contains NaN/Inf values.")
            # Preserve integer labels as ints when possible.
            if np.all(np.isclose(numeric_labels, np.round(numeric_labels))):
                label_array = np.asarray(np.round(numeric_labels), dtype=np.int64)
            else:
                label_array = numeric_labels

        return label_array

    def _coerce_score_array(self, value: object, name: str) -> np.ndarray:
        """Coerce probability/score payload to rank-1 or rank-2 float array."""
        try:
            score_array: np.ndarray = np.asarray(value, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            raise LinearProbeInputError(
                f"{name} cannot be converted to float numpy array: {exc}"
            ) from exc

        if score_array.ndim == 0:
            score_array = score_array.reshape(1)

        if score_array.ndim not in {1, 2}:
            raise LinearProbeInputError(
                f"{name} must be rank-1 or rank-2, got shape={tuple(score_array.shape)}."
            )

        if int(score_array.shape[0]) <= 0:
            raise LinearProbeInputError(f"{name} cannot be empty.")

        if not np.isfinite(score_array).all():
            raise LinearProbeInputError(f"{name} contains NaN/Inf values.")

        return score_array

    @staticmethod
    def _validate_c_value(value: float) -> float:
        if isinstance(value, bool):
            raise LinearProbeConfigError("c_value must be float, got bool.")
        try:
            c_value_float: float = float(value)
        except Exception as exc:  # noqa: BLE001
            raise LinearProbeConfigError(f"c_value must be float, got {value!r}.") from exc
        if not np.isfinite(c_value_float) or c_value_float <= 0.0:
            raise LinearProbeConfigError(f"c_value must be finite and > 0, got {c_value_float}.")
        return c_value_float

    @staticmethod
    def _validate_max_iter(value: int) -> int:
        if isinstance(value, bool):
            raise LinearProbeConfigError("max_iter must be int, got bool.")
        try:
            max_iter_int: int = int(value)
        except Exception as exc:  # noqa: BLE001
            raise LinearProbeConfigError(f"max_iter must be int, got {value!r}.") from exc
        if max_iter_int <= 0:
            raise LinearProbeConfigError(f"max_iter must be > 0, got {max_iter_int}.")
        return max_iter_int

    @staticmethod
    def _validate_solver(value: str) -> str:
        solver_str: str = "" if value is None else str(value).strip().lower()
        if solver_str == "":
            solver_str = DEFAULT_SOLVER
        if solver_str not in _ALLOWED_SOLVERS:
            raise LinearProbeConfigError(
                f"Unsupported solver={value!r}. Allowed: {_ALLOWED_SOLVERS}."
            )
        return solver_str

    @staticmethod
    def _validate_class_weight(value: str) -> str:
        class_weight_str: str = "" if value is None else str(value).strip().lower()
        if class_weight_str == "":
            class_weight_str = DEFAULT_CLASS_WEIGHT

        if class_weight_str != "balanced":
            raise LinearProbeConfigError(
                "class_weight must be 'balanced' for paper-aligned linear probing. "
                f"Got {value!r}."
            )
        return class_weight_str


__all__ = [
    "LinearProbeError",
    "LinearProbeConfigError",
    "LinearProbeInputError",
    "LinearProbeNotFittedError",
    "LinearProbeEvaluator",
]
