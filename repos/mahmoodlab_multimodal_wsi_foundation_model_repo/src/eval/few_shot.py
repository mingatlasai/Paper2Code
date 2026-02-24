"""Few-shot evaluation for TITAN slide embeddings.

This module implements the design-locked interface:
- Evaluator.run_few_shot(features: np.ndarray, y: np.ndarray, shots: list[int], runs: int = 50) -> dict

Implemented protocol (paper/config aligned):
- Shots: K in {1, 2, 4, 8, 16, 32}
- Runs: 50 repeated sampling runs
- Primary mode: SimpleShot prototype classification (Euclidean to class prototypes)
- Optional mode: few-shot linear probe (L2=1, max_iter=1000; no validation search)
- Preprocess per run: center + L2 normalize using support-only statistics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# -----------------------------------------------------------------------------
# Config-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_FEW_SHOT_SHOTS: Tuple[int, ...] = (1, 2, 4, 8, 16, 32)
_FEW_SHOT_RUNS: int = 50

_LINEAR_NO_VAL_L2: float = 1.0
_LINEAR_NO_VAL_MAX_ITER: int = 1000
_LINEAR_SOLVER: str = "lbfgs"

_DEFAULT_RANDOM_STATE: int = 42
_DEFAULT_EPS: float = 1.0e-12


class FewShotError(RuntimeError):
    """Base exception for few-shot evaluation failures."""


class FewShotSchemaError(FewShotError):
    """Raised when few-shot input schema/shape contracts are violated."""


@dataclass(frozen=True)
class _FoldSpec:
    """Single fold support/test index specification."""

    train_idx: np.ndarray
    test_idx: np.ndarray
    fold_id: str


@dataclass(frozen=True)
class _RunResult:
    """Single (fold, shot, run, mode) result container."""

    fold_id: str
    shot_k: int
    run_id: int
    mode: str
    support_size_total: int
    support_class_count: int
    test_size_total: int
    test_size_evaluated: int
    skipped_test_unseen_class: int
    metrics: Dict[str, float]


class FewShotEvaluator:
    """Few-shot evaluator with prototype and optional linear modes.

    Args:
        shots_default: Default shot list (must match config values).
        runs_default: Default number of runs (must match config value 50).
        embedding_service: Optional object exposing
            `center_and_l2(x, mean_vec=None) -> tuple[np.ndarray, np.ndarray]`.
        include_linear_probe: Whether to also run few-shot linear probe baseline.
        split: Optional split mapping. Supported forms:
            - {"train": idx_spec, "test": idx_spec}
            - {"folds": [{"train": ..., "test": ..., "fold_id": ...}, ...]}
          If omitted, a deterministic fallback is used:
            - train = all samples
            - test = all samples
            - per-run evaluation excludes sampled support indices.
        base_seed: Base random seed for deterministic run sampling.
    """

    def __init__(
        self,
        shots_default: Sequence[int] = _FEW_SHOT_SHOTS,
        runs_default: int = _FEW_SHOT_RUNS,
        embedding_service: Optional[Any] = None,
        include_linear_probe: bool = True,
        split: Optional[Mapping[str, Any]] = None,
        base_seed: int = _DEFAULT_RANDOM_STATE,
    ) -> None:
        if not isinstance(shots_default, Sequence) or isinstance(shots_default, (str, bytes)):
            raise TypeError("shots_default must be a sequence of integers.")

        shots_tuple: Tuple[int, ...] = tuple(int(v) for v in shots_default)
        if len(shots_tuple) == 0:
            raise ValueError("shots_default cannot be empty.")
        if any(v <= 0 for v in shots_tuple):
            raise ValueError("shots_default values must all be > 0.")
        if shots_tuple != _FEW_SHOT_SHOTS:
            raise ValueError(
                f"shots_default must be {list(_FEW_SHOT_SHOTS)} per config, got {list(shots_tuple)}."
            )

        if isinstance(runs_default, bool) or not isinstance(runs_default, int):
            raise TypeError("runs_default must be an integer.")
        if runs_default <= 0:
            raise ValueError("runs_default must be > 0.")
        if int(runs_default) != _FEW_SHOT_RUNS:
            raise ValueError(
                f"runs_default must be {_FEW_SHOT_RUNS} per config, got {runs_default}."
            )

        if embedding_service is not None:
            center_fn: Any = getattr(embedding_service, "center_and_l2", None)
            if not callable(center_fn):
                raise TypeError(
                    "embedding_service must implement callable center_and_l2(x, mean_vec=None)."
                )

        if isinstance(base_seed, bool) or not isinstance(base_seed, int):
            raise TypeError("base_seed must be an integer.")
        if base_seed < 0:
            raise ValueError("base_seed must be >= 0.")

        self.shots_default: Tuple[int, ...] = shots_tuple
        self.runs_default: int = int(runs_default)
        self.embedding_service: Optional[Any] = embedding_service
        self.include_linear_probe: bool = bool(include_linear_probe)
        self.split: Optional[Mapping[str, Any]] = split
        self.base_seed: int = int(base_seed)

        # Provenance constants.
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION
        self.feature_dim: int = _FEATURE_DIM
        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

        self.linear_no_val_l2: float = _LINEAR_NO_VAL_L2
        self.linear_no_val_max_iter: int = _LINEAR_NO_VAL_MAX_ITER

    def run_few_shot(
        self,
        features: np.ndarray,
        y: np.ndarray,
        shots: list[int],
        runs: int = _FEW_SHOT_RUNS,
    ) -> dict:
        """Run few-shot evaluation.

        Args:
            features: Embedding matrix [N, 768].
            y: Label vector [N].
            shots: Shot values to evaluate. Must match config default list.
            runs: Number of repeated runs. Must be 50 by default protocol.

        Returns:
            Structured few-shot result dictionary with run-level and aggregate metrics.
        """
        x: np.ndarray = self._validate_features(features)
        y_raw: np.ndarray = self._validate_labels(y=y, expected_n=int(x.shape[0]))

        if not isinstance(shots, list):
            raise TypeError(f"shots must be list[int], got {type(shots).__name__}.")
        shot_values: List[int] = [int(v) for v in shots]
        if len(shot_values) == 0:
            raise FewShotSchemaError("shots cannot be empty.")
        if any(v <= 0 for v in shot_values):
            raise FewShotSchemaError("shots values must all be > 0.")
        if tuple(shot_values) != self.shots_default:
            raise FewShotSchemaError(
                f"shots must be {list(self.shots_default)} per config, got {shot_values}."
            )

        if isinstance(runs, bool) or not isinstance(runs, int):
            raise TypeError("runs must be an integer.")
        if runs <= 0:
            raise FewShotSchemaError("runs must be > 0.")
        if int(runs) != self.runs_default:
            raise FewShotSchemaError(
                f"runs must be {self.runs_default} per config, got {runs}."
            )

        encoder: LabelEncoder = LabelEncoder()
        y_encoded: np.ndarray = encoder.fit_transform(y_raw)
        class_names: List[str] = [str(item) for item in encoder.classes_.tolist()]
        n_classes: int = int(len(class_names))
        if n_classes < 2:
            raise FewShotSchemaError("Few-shot evaluation requires at least 2 classes.")

        folds: List[_FoldSpec] = self._parse_folds(
            split=self.split,
            n_samples=int(x.shape[0]),
        )

        run_results: List[_RunResult] = []
        for fold_index, fold in enumerate(folds):
            x_train_pool: np.ndarray = x[fold.train_idx]
            y_train_pool: np.ndarray = y_encoded[fold.train_idx]
            x_test_full: np.ndarray = x[fold.test_idx]
            y_test_full: np.ndarray = y_encoded[fold.test_idx]

            if int(x_train_pool.shape[0]) <= 0:
                raise FewShotSchemaError(f"{fold.fold_id}: empty train pool.")
            if int(x_test_full.shape[0]) <= 0:
                raise FewShotSchemaError(f"{fold.fold_id}: empty test set.")

            classes_in_train: np.ndarray = np.unique(y_train_pool)
            if int(classes_in_train.shape[0]) < 2:
                raise FewShotSchemaError(
                    f"{fold.fold_id}: train pool has fewer than 2 classes."
                )

            class_to_pool_indices: Dict[int, np.ndarray] = {}
            for class_id in classes_in_train.tolist():
                class_mask: np.ndarray = y_train_pool == int(class_id)
                class_to_pool_indices[int(class_id)] = np.nonzero(class_mask)[0].astype(np.int64)

            for shot_k in shot_values:
                for run_id in range(int(runs)):
                    rng_seed: int = self._derive_run_seed(
                        base_seed=self.base_seed,
                        fold_index=fold_index,
                        shot_k=int(shot_k),
                        run_id=int(run_id),
                    )
                    rng: np.random.Generator = np.random.default_rng(rng_seed)

                    support_local_idx: np.ndarray = self._sample_support_indices_per_class(
                        class_to_pool_indices=class_to_pool_indices,
                        shot_k=int(shot_k),
                        rng=rng,
                    )
                    if int(support_local_idx.shape[0]) <= 0:
                        raise FewShotSchemaError(
                            f"{fold.fold_id}, shot={shot_k}, run={run_id}: sampled empty support set."
                        )

                    support_x: np.ndarray = x_train_pool[support_local_idx]
                    support_y: np.ndarray = y_train_pool[support_local_idx]

                    # Build evaluation test set for this run.
                    # If split is omitted, avoid direct support/test overlap.
                    if self.split is None:
                        support_global_idx: np.ndarray = fold.train_idx[support_local_idx]
                        support_global_set: set[int] = set(int(v) for v in support_global_idx.tolist())
                        test_keep_mask: np.ndarray = np.asarray(
                            [int(idx) not in support_global_set for idx in fold.test_idx.tolist()],
                            dtype=np.bool_,
                        )
                        x_test: np.ndarray = x_test_full[test_keep_mask]
                        y_test: np.ndarray = y_test_full[test_keep_mask]
                    else:
                        x_test = x_test_full
                        y_test = y_test_full

                    if int(x_test.shape[0]) <= 0:
                        # Skip impossible run (rare when split is omitted and support saturates).
                        continue

                    # Support-only centering and L2 normalization.
                    support_norm, mean_vec = self._center_and_l2(
                        x=support_x,
                        mean_vec=None,
                    )
                    test_norm, _ = self._center_and_l2(
                        x=x_test,
                        mean_vec=mean_vec,
                    )

                    # Keep only test samples whose class exists in sampled support.
                    support_classes_set: set[int] = set(int(v) for v in np.unique(support_y).tolist())
                    eval_mask: np.ndarray = np.asarray(
                        [int(lbl) in support_classes_set for lbl in y_test.tolist()],
                        dtype=np.bool_,
                    )

                    skipped_unseen: int = int((~eval_mask).sum())
                    if int(eval_mask.sum()) <= 0:
                        continue

                    test_eval_x: np.ndarray = test_norm[eval_mask]
                    test_eval_y: np.ndarray = y_test[eval_mask]

                    proto_result: _RunResult = self._evaluate_prototype_run(
                        fold_id=fold.fold_id,
                        shot_k=int(shot_k),
                        run_id=int(run_id),
                        support_x=support_norm,
                        support_y=support_y,
                        test_x=test_eval_x,
                        test_y=test_eval_y,
                        test_size_total=int(x_test.shape[0]),
                        skipped_unseen=skipped_unseen,
                    )
                    run_results.append(proto_result)

                    if self.include_linear_probe:
                        linear_result: _RunResult = self._evaluate_linear_run(
                            fold_id=fold.fold_id,
                            shot_k=int(shot_k),
                            run_id=int(run_id),
                            support_x=support_norm,
                            support_y=support_y,
                            test_x=test_eval_x,
                            test_y=test_eval_y,
                            test_size_total=int(x_test.shape[0]),
                            skipped_unseen=skipped_unseen,
                        )
                        run_results.append(linear_result)

        if len(run_results) == 0:
            raise FewShotSchemaError("No valid few-shot runs were produced.")

        aggregate_by_shot_mode: List[Dict[str, Any]] = self._aggregate_results(run_results)

        output: Dict[str, Any] = {
            "task": "few_shot",
            "input": {
                "n_samples": int(x.shape[0]),
                "n_features": int(x.shape[1]),
                "n_classes": int(n_classes),
                "classes": class_names,
                "num_folds": int(len(folds)),
            },
            "protocol": {
                "shots": [int(v) for v in shot_values],
                "runs": int(runs),
                "base_seed": int(self.base_seed),
                "primary_mode": "prototype",
                "include_linear_probe": bool(self.include_linear_probe),
                "linear_no_val_l2": float(self.linear_no_val_l2),
                "linear_no_val_max_iter": int(self.linear_no_val_max_iter),
                "split_provided": bool(self.split is not None),
            },
            "runs": [
                {
                    "fold_id": item.fold_id,
                    "shot_k": int(item.shot_k),
                    "run_id": int(item.run_id),
                    "mode": item.mode,
                    "support_size_total": int(item.support_size_total),
                    "support_class_count": int(item.support_class_count),
                    "test_size_total": int(item.test_size_total),
                    "test_size_evaluated": int(item.test_size_evaluated),
                    "skipped_test_unseen_class": int(item.skipped_test_unseen_class),
                    "metrics": dict(item.metrics),
                }
                for item in run_results
            ],
            "aggregate_by_shot_mode": aggregate_by_shot_mode,
        }
        return output

    def _evaluate_prototype_run(
        self,
        fold_id: str,
        shot_k: int,
        run_id: int,
        support_x: np.ndarray,
        support_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        test_size_total: int,
        skipped_unseen: int,
    ) -> _RunResult:
        support_classes: np.ndarray = np.unique(support_y)
        class_to_pos: Dict[int, int] = {
            int(class_id): int(pos) for pos, class_id in enumerate(support_classes.tolist())
        }

        prototypes: np.ndarray = np.zeros((int(support_classes.shape[0]), int(support_x.shape[1])), dtype=np.float64)
        for class_id in support_classes.tolist():
            cls_mask: np.ndarray = support_y == int(class_id)
            prototypes[int(class_to_pos[int(class_id)])] = np.mean(support_x[cls_mask], axis=0, dtype=np.float64)

        # Normalize prototypes for stable distances.
        proto_norms: np.ndarray = np.linalg.norm(prototypes, ord=2, axis=1, keepdims=True)
        proto_norms = np.maximum(proto_norms, float(_DEFAULT_EPS))
        prototypes = prototypes / proto_norms

        distances: np.ndarray = self._pairwise_euclidean(test_x.astype(np.float64), prototypes)
        pred_pos: np.ndarray = np.argmin(distances, axis=1)
        y_pred: np.ndarray = np.asarray([int(support_classes[int(pos)]) for pos in pred_pos.tolist()], dtype=np.int64)

        # Probability-like scores via softmax(-distance), then expanded to global class ids.
        logits: np.ndarray = -distances
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits: np.ndarray = np.exp(logits)
        prob_local: np.ndarray = exp_logits / np.maximum(np.sum(exp_logits, axis=1, keepdims=True), _DEFAULT_EPS)

        n_classes_global: int = int(np.max(np.concatenate([support_y, test_y])) + 1)
        y_score_global: np.ndarray = np.zeros((int(test_x.shape[0]), n_classes_global), dtype=np.float64)
        for pos, class_id in enumerate(support_classes.tolist()):
            y_score_global[:, int(class_id)] = prob_local[:, int(pos)]

        metrics: Dict[str, float] = self._compute_metrics(
            y_true=test_y,
            y_pred=y_pred,
            y_score=y_score_global,
        )

        return _RunResult(
            fold_id=str(fold_id),
            shot_k=int(shot_k),
            run_id=int(run_id),
            mode="prototype",
            support_size_total=int(support_x.shape[0]),
            support_class_count=int(support_classes.shape[0]),
            test_size_total=int(test_size_total),
            test_size_evaluated=int(test_x.shape[0]),
            skipped_test_unseen_class=int(skipped_unseen),
            metrics=metrics,
        )

    def _evaluate_linear_run(
        self,
        fold_id: str,
        shot_k: int,
        run_id: int,
        support_x: np.ndarray,
        support_y: np.ndarray,
        test_x: np.ndarray,
        test_y: np.ndarray,
        test_size_total: int,
        skipped_unseen: int,
    ) -> _RunResult:
        support_classes: np.ndarray = np.unique(support_y)
        if int(support_classes.shape[0]) < 2:
            raise FewShotSchemaError(
                f"{fold_id}, shot={shot_k}, run={run_id}: linear mode requires >=2 support classes."
            )

        c_value: float = 1.0 / float(self.linear_no_val_l2)
        model: LogisticRegression = LogisticRegression(
            penalty="l2",
            C=float(c_value),
            solver=_LINEAR_SOLVER,
            max_iter=int(self.linear_no_val_max_iter),
            multi_class="auto",
            random_state=int(self.base_seed),
        )
        model.fit(support_x, support_y)

        y_pred: np.ndarray = model.predict(test_x)
        y_prob_local: np.ndarray = model.predict_proba(test_x)

        model_classes: np.ndarray = np.asarray(model.classes_, dtype=np.int64)
        n_classes_global: int = int(max(np.max(support_y), np.max(test_y)) + 1)
        y_score_global: np.ndarray = np.zeros((int(test_x.shape[0]), n_classes_global), dtype=np.float64)
        for idx, class_id in enumerate(model_classes.tolist()):
            y_score_global[:, int(class_id)] = y_prob_local[:, int(idx)]

        metrics: Dict[str, float] = self._compute_metrics(
            y_true=test_y,
            y_pred=y_pred,
            y_score=y_score_global,
        )

        return _RunResult(
            fold_id=str(fold_id),
            shot_k=int(shot_k),
            run_id=int(run_id),
            mode="linear",
            support_size_total=int(support_x.shape[0]),
            support_class_count=int(support_classes.shape[0]),
            test_size_total=int(test_size_total),
            test_size_evaluated=int(test_x.shape[0]),
            skipped_test_unseen_class=int(skipped_unseen),
            metrics=metrics,
        )

    def _sample_support_indices_per_class(
        self,
        class_to_pool_indices: Mapping[int, np.ndarray],
        shot_k: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        sampled_chunks: List[np.ndarray] = []
        for class_id in sorted(class_to_pool_indices.keys()):
            pool_idx: np.ndarray = np.asarray(class_to_pool_indices[class_id], dtype=np.int64)
            if pool_idx.ndim != 1:
                raise FewShotSchemaError(
                    f"Class {class_id} pool indices must be rank-1, got {tuple(pool_idx.shape)}."
                )
            if int(pool_idx.shape[0]) <= 0:
                continue

            sample_count: int = min(int(shot_k), int(pool_idx.shape[0]))
            selected: np.ndarray = rng.choice(pool_idx, size=sample_count, replace=False)
            sampled_chunks.append(np.asarray(selected, dtype=np.int64))

        if len(sampled_chunks) == 0:
            return np.zeros((0,), dtype=np.int64)

        support_idx: np.ndarray = np.concatenate(sampled_chunks, axis=0)
        support_idx = np.unique(support_idx)
        return support_idx

    def _aggregate_results(self, run_results: Sequence[_RunResult]) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[int, str], List[_RunResult]] = {}
        for item in run_results:
            key: Tuple[int, str] = (int(item.shot_k), str(item.mode))
            grouped.setdefault(key, []).append(item)

        aggregates: List[Dict[str, Any]] = []
        for key in sorted(grouped.keys(), key=lambda pair: (pair[0], pair[1])):
            shot_k, mode = key
            items: List[_RunResult] = grouped[key]

            metric_names: List[str] = sorted({name for item in items for name in item.metrics.keys()})
            metric_summary: Dict[str, float] = {}
            for metric_name in metric_names:
                values: List[float] = [
                    float(item.metrics[metric_name])
                    for item in items
                    if metric_name in item.metrics and np.isfinite(float(item.metrics[metric_name]))
                ]
                if len(values) == 0:
                    continue
                arr: np.ndarray = np.asarray(values, dtype=np.float64)
                metric_summary[f"{metric_name}_mean"] = float(np.mean(arr))
                if int(arr.shape[0]) > 1:
                    metric_summary[f"{metric_name}_std"] = float(np.std(arr, ddof=1))
                else:
                    metric_summary[f"{metric_name}_std"] = 0.0

            support_sizes: np.ndarray = np.asarray([item.support_size_total for item in items], dtype=np.float64)
            test_sizes_eval: np.ndarray = np.asarray([item.test_size_evaluated for item in items], dtype=np.float64)
            skipped_counts: np.ndarray = np.asarray([item.skipped_test_unseen_class for item in items], dtype=np.float64)

            aggregates.append(
                {
                    "shot_k": int(shot_k),
                    "mode": str(mode),
                    "completed_runs": int(len(items)),
                    "support_size_mean": float(np.mean(support_sizes)),
                    "support_size_std": float(np.std(support_sizes, ddof=1)) if int(len(items)) > 1 else 0.0,
                    "test_size_evaluated_mean": float(np.mean(test_sizes_eval)),
                    "test_size_evaluated_std": float(np.std(test_sizes_eval, ddof=1)) if int(len(items)) > 1 else 0.0,
                    "skipped_test_unseen_class_mean": float(np.mean(skipped_counts)),
                    "metrics": metric_summary,
                }
            )

        return aggregates

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_score: Optional[np.ndarray]) -> Dict[str, float]:
        if y_true.ndim != 1 or y_pred.ndim != 1:
            raise FewShotSchemaError(
                f"y_true and y_pred must be rank-1, got {tuple(y_true.shape)} and {tuple(y_pred.shape)}."
            )
        if int(y_true.shape[0]) != int(y_pred.shape[0]):
            raise FewShotSchemaError(
                f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}."
            )
        if int(y_true.shape[0]) <= 0:
            raise FewShotSchemaError("Cannot compute metrics on empty vectors.")

        metrics: Dict[str, float] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        }

        unique_classes: np.ndarray = np.unique(y_true)
        n_classes_true: int = int(unique_classes.shape[0])
        if n_classes_true <= 1:
            raise FewShotSchemaError("y_true must include at least 2 classes for metric computation.")

        if n_classes_true == 2:
            if y_score is not None:
                if y_score.ndim != 2:
                    raise FewShotSchemaError(
                        f"Binary y_score must be rank-2 [N,C], got {tuple(y_score.shape)}."
                    )
                # Positive class = larger encoded class id among classes present in y_true.
                pos_class: int = int(np.max(unique_classes))
                if int(y_score.shape[1]) <= pos_class:
                    raise FewShotSchemaError(
                        "y_score does not include positive-class column index. "
                        f"shape={tuple(y_score.shape)}, pos_class={pos_class}."
                    )
                pos_scores: np.ndarray = y_score[:, pos_class]
                metrics["auroc"] = float(roc_auc_score(y_true, pos_scores))
        else:
            metrics["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted"))

        return metrics

    def _center_and_l2(self, x: np.ndarray, mean_vec: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        if self.embedding_service is not None:
            centered, mean_used = self.embedding_service.center_and_l2(x=x, mean_vec=mean_vec)
            centered_np: np.ndarray = np.asarray(centered, dtype=np.float32)
            mean_np: np.ndarray = np.asarray(mean_used, dtype=np.float32)
            if centered_np.shape != x.shape:
                raise FewShotSchemaError(
                    "embedding_service.center_and_l2 returned invalid shape. "
                    f"Expected {x.shape}, got {centered_np.shape}."
                )
            if mean_np.ndim != 1 or int(mean_np.shape[0]) != int(x.shape[1]):
                raise FewShotSchemaError(
                    "embedding_service.center_and_l2 returned invalid mean shape. "
                    f"Expected ({x.shape[1]},), got {mean_np.shape}."
                )
            return centered_np, mean_np

        matrix: np.ndarray = np.asarray(x, dtype=np.float32)
        if matrix.ndim != 2:
            raise FewShotSchemaError(
                f"center_and_l2 expects [N,D], got {tuple(matrix.shape)}."
            )
        if int(matrix.shape[0]) <= 0:
            raise FewShotSchemaError("center_and_l2 received empty matrix.")
        if not np.isfinite(matrix).all():
            raise FewShotSchemaError("center_and_l2 input contains NaN/Inf.")

        if mean_vec is None:
            mean_local: np.ndarray = np.mean(matrix, axis=0, dtype=np.float64).astype(np.float32, copy=False)
        else:
            mean_local = np.asarray(mean_vec, dtype=np.float32)
            if mean_local.ndim != 1 or int(mean_local.shape[0]) != int(matrix.shape[1]):
                raise FewShotSchemaError(
                    "mean_vec must have shape [D]. "
                    f"Expected ({matrix.shape[1]},), got {mean_local.shape}."
                )

        centered: np.ndarray = matrix - mean_local.reshape(1, -1)
        norms: np.ndarray = np.linalg.norm(centered, ord=2, axis=1, keepdims=True)
        norms = np.maximum(norms, float(_DEFAULT_EPS)).astype(np.float32, copy=False)
        normalized: np.ndarray = centered / norms

        if not np.isfinite(normalized).all():
            raise FewShotSchemaError("center_and_l2 produced NaN/Inf values.")

        return normalized.astype(np.float32, copy=False), mean_local.astype(np.float32, copy=False)

    @staticmethod
    def _pairwise_euclidean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if a.ndim != 2 or b.ndim != 2:
            raise FewShotSchemaError(
                f"pairwise_euclidean expects rank-2 arrays, got {a.shape} and {b.shape}."
            )
        if int(a.shape[1]) != int(b.shape[1]):
            raise FewShotSchemaError(
                "pairwise_euclidean feature dimension mismatch: "
                f"{a.shape[1]} vs {b.shape[1]}."
            )

        a2: np.ndarray = np.sum(a * a, axis=1, keepdims=True)
        b2: np.ndarray = np.sum(b * b, axis=1, keepdims=True).T
        sq: np.ndarray = a2 + b2 - (2.0 * np.matmul(a, b.T))
        sq = np.maximum(sq, 0.0)
        dist: np.ndarray = np.sqrt(sq, dtype=np.float64)
        return dist

    @staticmethod
    def _validate_features(features: Any) -> np.ndarray:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features)
        x: np.ndarray = np.asarray(features, dtype=np.float64)

        if x.ndim != 2:
            raise FewShotSchemaError(f"features must be rank-2 [N,D], got {tuple(x.shape)}.")
        if int(x.shape[0]) <= 1:
            raise FewShotSchemaError("features must contain at least 2 samples.")
        if int(x.shape[1]) != _FEATURE_DIM:
            raise FewShotSchemaError(
                f"features second dimension must be {_FEATURE_DIM}, got {int(x.shape[1])}."
            )
        if not np.isfinite(x).all():
            raise FewShotSchemaError("features contain NaN/Inf values.")
        return x

    @staticmethod
    def _validate_labels(y: Any, expected_n: int) -> np.ndarray:
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        labels: np.ndarray = np.asarray(y)

        if labels.ndim != 1:
            raise FewShotSchemaError(f"y must be rank-1 [N], got {tuple(labels.shape)}.")
        if int(labels.shape[0]) != int(expected_n):
            raise FewShotSchemaError(
                f"y length must match number of samples ({expected_n}), got {int(labels.shape[0])}."
            )
        if int(labels.shape[0]) <= 1:
            raise FewShotSchemaError("y must contain at least 2 samples.")
        return labels

    def _parse_folds(self, split: Optional[Mapping[str, Any]], n_samples: int) -> List[_FoldSpec]:
        if split is None:
            all_idx: np.ndarray = np.arange(n_samples, dtype=np.int64)
            return [_FoldSpec(train_idx=all_idx, test_idx=all_idx, fold_id="fold_0")]

        if not isinstance(split, Mapping):
            raise FewShotSchemaError(f"split must be a mapping, got {type(split).__name__}.")

        if "folds" in split:
            folds_obj: Any = split.get("folds")
            if not isinstance(folds_obj, Sequence) or isinstance(folds_obj, (str, bytes)):
                raise FewShotSchemaError("split['folds'] must be a sequence of fold mappings.")

            fold_specs: List[_FoldSpec] = []
            for fold_idx, fold_item in enumerate(folds_obj):
                if not isinstance(fold_item, Mapping):
                    raise FewShotSchemaError(
                        f"Each fold must be mapping, got {type(fold_item).__name__} at index {fold_idx}."
                    )
                fold_id: str = str(fold_item.get("fold_id", f"fold_{fold_idx}"))
                fold_specs.append(
                    self._parse_single_fold(
                        split=fold_item,
                        n_samples=n_samples,
                        fold_id=fold_id,
                    )
                )
            if len(fold_specs) == 0:
                raise FewShotSchemaError("split['folds'] is empty.")
            return fold_specs

        return [
            self._parse_single_fold(
                split=split,
                n_samples=n_samples,
                fold_id="fold_0",
            )
        ]

    def _parse_single_fold(self, split: Mapping[str, Any], n_samples: int, fold_id: str) -> _FoldSpec:
        if "train" not in split or "test" not in split:
            raise FewShotSchemaError(
                "Each fold split must include 'train' and 'test' index specs."
            )

        train_idx: np.ndarray = self._to_indices(split["train"], n_samples=n_samples, name=f"{fold_id}.train")
        test_idx: np.ndarray = self._to_indices(split["test"], n_samples=n_samples, name=f"{fold_id}.test")

        if int(train_idx.shape[0]) <= 0:
            raise FewShotSchemaError(f"{fold_id}: train split is empty.")
        if int(test_idx.shape[0]) <= 0:
            raise FewShotSchemaError(f"{fold_id}: test split is empty.")

        return _FoldSpec(train_idx=train_idx, test_idx=test_idx, fold_id=str(fold_id))

    @staticmethod
    def _to_indices(spec: Any, n_samples: int, name: str) -> np.ndarray:
        if isinstance(spec, np.ndarray):
            arr: np.ndarray = spec
        elif isinstance(spec, (list, tuple)):
            arr = np.asarray(spec)
        else:
            raise FewShotSchemaError(
                f"{name}: index spec must be list/tuple/np.ndarray, got {type(spec).__name__}."
            )

        if arr.ndim != 1:
            raise FewShotSchemaError(f"{name}: index spec must be rank-1, got {tuple(arr.shape)}.")

        if arr.dtype == np.bool_:
            if int(arr.shape[0]) != int(n_samples):
                raise FewShotSchemaError(
                    f"{name}: boolean mask length must be {n_samples}, got {int(arr.shape[0])}."
                )
            indices: np.ndarray = np.nonzero(arr)[0].astype(np.int64, copy=False)
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                if np.issubdtype(arr.dtype, np.floating):
                    if not np.all(np.equal(arr, np.floor(arr))):
                        raise FewShotSchemaError(f"{name}: floating indices must be integer-valued.")
                    arr = arr.astype(np.int64)
                else:
                    raise FewShotSchemaError(
                        f"{name}: indices must be integer or bool mask, got dtype={arr.dtype}."
                    )
            indices = np.asarray(arr, dtype=np.int64)

        if indices.size == 0:
            return indices

        if np.any(indices < 0) or np.any(indices >= int(n_samples)):
            raise FewShotSchemaError(
                f"{name}: indices out of range [0,{n_samples - 1}]."
            )

        unique_indices: np.ndarray = np.unique(indices)
        if int(unique_indices.shape[0]) != int(indices.shape[0]):
            raise FewShotSchemaError(f"{name}: duplicate indices are not allowed.")

        return unique_indices

    @staticmethod
    def _derive_run_seed(base_seed: int, fold_index: int, shot_k: int, run_id: int) -> int:
        # Deterministic, collision-resistant integer mix in 32-bit range.
        seed_value: int = int(base_seed)
        seed_value = (seed_value * 1_000_003 + int(fold_index) * 97_409) % (2**32 - 1)
        seed_value = (seed_value * 1_000_003 + int(shot_k) * 65_537) % (2**32 - 1)
        seed_value = (seed_value * 1_000_003 + int(run_id) * 8_191) % (2**32 - 1)
        if seed_value <= 0:
            seed_value = 1
        return seed_value


# Convenience functional API.
def run_few_shot(
    features: np.ndarray,
    y: np.ndarray,
    shots: list[int],
    runs: int = _FEW_SHOT_RUNS,
) -> dict:
    """Run few-shot evaluation with default config-aligned evaluator."""
    evaluator: FewShotEvaluator = FewShotEvaluator()
    return evaluator.run_few_shot(features=features, y=y, shots=shots, runs=runs)


__all__ = [
    "FewShotError",
    "FewShotSchemaError",
    "FewShotEvaluator",
    "run_few_shot",
]
