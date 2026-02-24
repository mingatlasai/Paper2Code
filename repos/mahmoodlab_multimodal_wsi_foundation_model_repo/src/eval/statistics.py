"""Statistical utilities for TITAN evaluation reporting.

This module implements the design-locked interface:
- StatsAnalyzer.bootstrap_metrics(y_true, y_pred, n_boot=1000) -> dict
- StatsAnalyzer.fold_mean_std(scores: list[float]) -> dict
- StatsAnalyzer.fit_glmm(results_df: pd.DataFrame) -> Any
- StatsAnalyzer.pairwise_tests_glmm(model: Any, method: str = "tukey") -> pd.DataFrame

Paper-aligned behavior:
- Single-split uncertainty via nonparametric bootstrap (default n=1000).
- Fold-based reporting via mean +/- sample standard deviation.
- Optional mixed-effects analysis for bounded metrics using a practical,
  explicit approximation in statsmodels:
  - score in (0,1) is boundary-clipped and logit-transformed,
  - MixedLM with method as fixed effect and dataset as random intercept.
- Pairwise comparisons expose Tukey HSD on transformed scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import balanced_accuracy_score

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
except Exception:  # pragma: no cover - optional dependency at runtime
    smf = None
    pairwise_tukeyhsd = None


# -----------------------------------------------------------------------------
# Config-locked constants (from provided config/eval contracts).
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: tuple[int, int] = (64, 64)

_BOOTSTRAP_SAMPLES: int = 1000
_DEFAULT_BOOTSTRAP_SEED: int = 42
_DEFAULT_CI_ALPHA: float = 0.05

_DEFAULT_METHOD_COL: str = "method"
_DEFAULT_DATASET_COL: str = "dataset"
_DEFAULT_METRIC_COL: str = "score"
_DEFAULT_BOUNDARY_EPS: float = 1.0e-6


class StatsError(RuntimeError):
    """Base exception for statistical analysis failures."""


class StatsSchemaError(StatsError):
    """Raised when statistical input schema/shape is invalid."""


class StatsDependencyError(StatsError):
    """Raised when optional dependencies required by a method are unavailable."""


@dataclass(frozen=True)
class GLMMFitResult:
    """Container for fitted mixed-model artifacts.

    Attributes:
        backend: Fitting backend identifier.
        method_col: Method column name used for fixed effect.
        dataset_col: Dataset column name used for random intercept.
        metric_col: Original metric column in [0,1].
        transformed_col: Transformed metric column used for fitting.
        model_formula: Model formula used in statsmodels.
        converged: Whether model converged.
        fit_error: Optional fit error details when fitting fails.
        model_result: Raw statsmodels fitted result object (if available).
        prepared_df: Prepared long-form dataframe used for fitting.
    """

    backend: str
    method_col: str
    dataset_col: str
    metric_col: str
    transformed_col: str
    model_formula: str
    converged: bool
    fit_error: Optional[str]
    model_result: Any
    prepared_df: pd.DataFrame


class StatsAnalyzer:
    """Statistical analyzer for downstream evaluation summaries."""

    def __init__(
        self,
        bootstrap_default: int = _BOOTSTRAP_SAMPLES,
        bootstrap_seed: int = _DEFAULT_BOOTSTRAP_SEED,
        ci_alpha: float = _DEFAULT_CI_ALPHA,
        method_col: str = _DEFAULT_METHOD_COL,
        dataset_col: str = _DEFAULT_DATASET_COL,
        metric_col: str = _DEFAULT_METRIC_COL,
        boundary_eps: float = _DEFAULT_BOUNDARY_EPS,
    ) -> None:
        if isinstance(bootstrap_default, bool) or not isinstance(bootstrap_default, int):
            raise TypeError("bootstrap_default must be an integer.")
        if bootstrap_default <= 0:
            raise ValueError("bootstrap_default must be > 0.")
        if int(bootstrap_default) != _BOOTSTRAP_SAMPLES:
            raise ValueError(
                f"bootstrap_default must be {_BOOTSTRAP_SAMPLES} per config, got {bootstrap_default}."
            )

        if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
            raise TypeError("bootstrap_seed must be an integer.")
        if bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be >= 0.")

        if not isinstance(ci_alpha, (float, int)):
            raise TypeError("ci_alpha must be numeric.")
        ci_alpha_value: float = float(ci_alpha)
        if not (0.0 < ci_alpha_value < 1.0):
            raise ValueError("ci_alpha must be in (0, 1).")

        if not isinstance(method_col, str) or not method_col.strip():
            raise ValueError("method_col must be a non-empty string.")
        if not isinstance(dataset_col, str) or not dataset_col.strip():
            raise ValueError("dataset_col must be a non-empty string.")
        if not isinstance(metric_col, str) or not metric_col.strip():
            raise ValueError("metric_col must be a non-empty string.")

        if not isinstance(boundary_eps, (float, int)):
            raise TypeError("boundary_eps must be numeric.")
        boundary_eps_value: float = float(boundary_eps)
        if not (0.0 < boundary_eps_value < 0.5):
            raise ValueError("boundary_eps must be in (0, 0.5).")

        self.bootstrap_default: int = int(bootstrap_default)
        self.bootstrap_seed: int = int(bootstrap_seed)
        self.ci_alpha: float = ci_alpha_value

        self.method_col: str = method_col.strip()
        self.dataset_col: str = dataset_col.strip()
        self.metric_col: str = metric_col.strip()
        self.boundary_eps: float = boundary_eps_value

        # Provenance constants used across eval modules.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: tuple[int, int] = _STAGE3_CROP_GRID

    def bootstrap_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_boot: int = _BOOTSTRAP_SAMPLES,
    ) -> dict:
        """Estimate uncertainty via bootstrap resampling.

        This design-locked API computes a balanced-accuracy distribution using
        paired index-resampling over (y_true, y_pred).
        """
        y_true_arr: np.ndarray = self._to_1d_array(y_true, name="y_true")
        y_pred_arr: np.ndarray = self._to_1d_array(y_pred, name="y_pred")

        if int(y_true_arr.shape[0]) != int(y_pred_arr.shape[0]):
            raise StatsSchemaError(
                "y_true and y_pred must have equal length, got "
                f"{y_true_arr.shape[0]} and {y_pred_arr.shape[0]}."
            )
        if int(y_true_arr.shape[0]) <= 0:
            raise StatsSchemaError("y_true/y_pred must be non-empty.")

        if isinstance(n_boot, bool) or not isinstance(n_boot, int):
            raise TypeError("n_boot must be an integer.")
        if n_boot <= 0:
            raise ValueError("n_boot must be > 0.")

        # Strongly typed local aliases.
        n_samples: int = int(y_true_arr.shape[0])
        n_bootstrap: int = int(n_boot)
        ci_lower_q: float = 100.0 * (self.ci_alpha / 2.0)
        ci_upper_q: float = 100.0 * (1.0 - (self.ci_alpha / 2.0))

        point_estimate: float = float(balanced_accuracy_score(y_true_arr, y_pred_arr))

        rng: np.random.Generator = np.random.default_rng(self.bootstrap_seed)
        boot_values: np.ndarray = np.zeros((n_bootstrap,), dtype=np.float64)

        for boot_idx in range(n_bootstrap):
            sample_idx: np.ndarray = rng.integers(
                low=0,
                high=n_samples,
                size=n_samples,
                endpoint=False,
                dtype=np.int64,
            )
            y_true_resampled: np.ndarray = y_true_arr[sample_idx]
            y_pred_resampled: np.ndarray = y_pred_arr[sample_idx]
            boot_values[boot_idx] = float(
                balanced_accuracy_score(y_true_resampled, y_pred_resampled)
            )

        output: Dict[str, Any] = {
            "metric_name": "balanced_accuracy",
            "n_samples": n_samples,
            "n_boot": n_bootstrap,
            "seed": int(self.bootstrap_seed),
            "ci_alpha": float(self.ci_alpha),
            "ci_method": "percentile",
            "point_estimate": point_estimate,
            "boot_mean": float(np.mean(boot_values)),
            "boot_std": float(np.std(boot_values, ddof=1) if n_bootstrap > 1 else 0.0),
            "ci_lower": float(np.percentile(boot_values, ci_lower_q)),
            "ci_upper": float(np.percentile(boot_values, ci_upper_q)),
        }
        return output

    def fold_mean_std(self, scores: list[float]) -> dict:
        """Aggregate fold scores as mean and sample standard deviation."""
        if not isinstance(scores, list):
            raise TypeError(f"scores must be list[float], got {type(scores).__name__}.")
        if len(scores) <= 0:
            raise StatsSchemaError("scores cannot be empty.")

        arr: np.ndarray = np.asarray(scores, dtype=np.float64)
        if arr.ndim != 1:
            raise StatsSchemaError(f"scores must be rank-1, got shape={tuple(arr.shape)}.")
        if not np.isfinite(arr).all():
            raise StatsSchemaError("scores contain NaN/Inf values.")

        n_folds: int = int(arr.shape[0])
        output: Dict[str, Any] = {
            "n_folds": n_folds,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1) if n_folds > 1 else 0.0),
        }
        return output

    def fit_glmm(self, results_df: pd.DataFrame) -> Any:
        """Fit a practical mixed-effects model for bounded metrics.

        Implementation note:
        - Beta-family GLMM is not directly available in statsmodels 0.14.1 with
          random effects. This method uses a documented approximation:
          logit-transform bounded metric after boundary clipping, then fit MixedLM.
        """
        prepared_df: pd.DataFrame = self._prepare_glmm_frame(results_df)

        if smf is None:
            raise StatsDependencyError(
                "statsmodels is required for fit_glmm but could not be imported."
            )

        transformed_col: str = f"{self.metric_col}_logit"
        formula: str = f"{transformed_col} ~ C({self.method_col})"

        fit_error: Optional[str] = None
        model_result: Any = None
        converged: bool = False

        try:
            mixed_model = smf.mixedlm(
                formula=formula,
                data=prepared_df,
                groups=prepared_df[self.dataset_col],
                re_formula="1",
            )
            model_result = mixed_model.fit(reml=False, method="lbfgs", maxiter=500, disp=False)
            converged = bool(getattr(model_result, "converged", False))
        except Exception as exc:  # noqa: BLE001
            fit_error = str(exc)

        return GLMMFitResult(
            backend="statsmodels_mixedlm_logit",
            method_col=self.method_col,
            dataset_col=self.dataset_col,
            metric_col=self.metric_col,
            transformed_col=transformed_col,
            model_formula=formula,
            converged=converged,
            fit_error=fit_error,
            model_result=model_result,
            prepared_df=prepared_df,
        )

    def pairwise_tests_glmm(self, model: Any, method: str = "tukey") -> pd.DataFrame:
        """Run pairwise post-hoc tests from fitted GLMM context.

        Supported method:
        - ``method='tukey'``: Tukey HSD on transformed scores grouped by method.
        """
        method_name: str = str(method).strip().lower()
        if method_name != "tukey":
            raise ValueError(f"Unsupported pairwise method '{method}'. Only 'tukey' is supported.")

        if pairwise_tukeyhsd is None:
            raise StatsDependencyError(
                "statsmodels pairwise Tukey HSD is unavailable. Install statsmodels."
            )

        if not isinstance(model, GLMMFitResult):
            raise TypeError(
                "model must be GLMMFitResult returned by fit_glmm, "
                f"got {type(model).__name__}."
            )

        df: pd.DataFrame = model.prepared_df
        if df.empty:
            raise StatsSchemaError("model.prepared_df is empty; cannot run pairwise tests.")

        tukey_obj = pairwise_tukeyhsd(
            endog=df[model.transformed_col].to_numpy(dtype=np.float64),
            groups=df[model.method_col].astype(str).to_numpy(),
            alpha=float(self.ci_alpha),
        )

        # statsmodels returns a SimpleTable in summary(); pull machine-readable arrays.
        res: Any = tukey_obj._results_table  # pylint: disable=protected-access
        rows: List[List[Any]] = res.data[1:]

        out_rows: List[Dict[str, Any]] = []
        for row in rows:
            # row layout: group1, group2, meandiff, p-adj, lower, upper, reject
            group1: str = str(row[0])
            group2: str = str(row[1])
            meandiff: float = float(row[2])
            p_adj: float = float(row[3])
            lower: float = float(row[4])
            upper: float = float(row[5])
            reject: bool = bool(row[6])

            out_rows.append(
                {
                    "method_a": group1,
                    "method_b": group2,
                    "estimate": meandiff,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "p_value": p_adj,
                    "p_adj": p_adj,
                    "reject": reject,
                    "correction": "tukey",
                    "scale": "logit_transformed_metric",
                    "backend": model.backend,
                    "glmm_converged": bool(model.converged),
                }
            )

        output_df: pd.DataFrame = pd.DataFrame(out_rows)
        if output_df.empty:
            # Keep a stable schema even when no pairs are available.
            output_df = pd.DataFrame(
                columns=[
                    "method_a",
                    "method_b",
                    "estimate",
                    "ci_lower",
                    "ci_upper",
                    "p_value",
                    "p_adj",
                    "reject",
                    "correction",
                    "scale",
                    "backend",
                    "glmm_converged",
                ]
            )
        return output_df

    def _prepare_glmm_frame(self, results_df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(results_df, pd.DataFrame):
            raise TypeError(
                f"results_df must be pandas.DataFrame, got {type(results_df).__name__}."
            )
        if results_df.empty:
            raise StatsSchemaError("results_df is empty.")

        required_cols: List[str] = [self.method_col, self.dataset_col, self.metric_col]
        missing_cols: List[str] = [name for name in required_cols if name not in results_df.columns]
        if missing_cols:
            raise StatsSchemaError(
                f"results_df missing required columns: {missing_cols}. "
                f"Required={required_cols}."
            )

        df: pd.DataFrame = results_df.loc[:, required_cols].copy()

        # Enforce explicit dtypes and finite numeric metric.
        df[self.method_col] = df[self.method_col].astype(str)
        df[self.dataset_col] = df[self.dataset_col].astype(str)
        df[self.metric_col] = pd.to_numeric(df[self.metric_col], errors="coerce")

        if df[self.metric_col].isna().any():
            raise StatsSchemaError(
                f"Column '{self.metric_col}' contains non-numeric or NaN values."
            )

        metric_values: np.ndarray = df[self.metric_col].to_numpy(dtype=np.float64)
        if not np.isfinite(metric_values).all():
            raise StatsSchemaError(
                f"Column '{self.metric_col}' contains Inf/-Inf values."
            )
        if np.any(metric_values < 0.0) or np.any(metric_values > 1.0):
            raise StatsSchemaError(
                f"Column '{self.metric_col}' must be bounded in [0,1]."
            )

        # Boundary stabilization for logit transform.
        clipped: np.ndarray = np.clip(metric_values, self.boundary_eps, 1.0 - self.boundary_eps)
        transformed_col: str = f"{self.metric_col}_logit"
        df[transformed_col] = logit(clipped)

        if not np.isfinite(df[transformed_col].to_numpy(dtype=np.float64)).all():
            raise StatsSchemaError(
                f"Transformed column '{transformed_col}' contains non-finite values."
            )

        if int(df[self.method_col].nunique()) < 2:
            raise StatsSchemaError("GLMM requires at least 2 unique methods.")
        if int(df[self.dataset_col].nunique()) < 2:
            raise StatsSchemaError("GLMM requires at least 2 unique datasets/groups.")

        return df

    @staticmethod
    def _to_1d_array(value: np.ndarray, name: str) -> np.ndarray:
        arr: np.ndarray = np.asarray(value)
        if arr.ndim != 1:
            raise StatsSchemaError(f"{name} must be rank-1, got shape={tuple(arr.shape)}.")
        if int(arr.shape[0]) <= 0:
            raise StatsSchemaError(f"{name} cannot be empty.")
        return arr


__all__ = [
    "StatsError",
    "StatsSchemaError",
    "StatsDependencyError",
    "GLMMFitResult",
    "StatsAnalyzer",
]
