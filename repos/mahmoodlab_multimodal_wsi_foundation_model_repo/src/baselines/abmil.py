"""ABMIL baseline for supervised slide-level classification.

This module implements an attention-based MIL baseline aligned with the
paper/task design:
- Input patch feature dim: 768.
- Patch size: 512.
- Magnification: 20x.
- Default supervised training protocol:
  - batch_size = 1
  - optimizer = AdamW
  - lr = 1e-4
  - weight_decay = 1e-5
  - scheduler = CosineAnnealingLR
  - epochs = 20

Public API is intentionally simple and reusable:
- ABMILModel: torch module for bag-level logits and pooled embedding.
- ABMILBaseline: train/infer wrapper with deterministic contracts.
- run_abmil: convenience function for one-shot train + evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# -----------------------------------------------------------------------------
# Config-locked constants from provided project configuration.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768
_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_CROP_GRID: Tuple[int, int] = (64, 64)

_DEFAULT_BATCH_SIZE: int = 1
_DEFAULT_EPOCHS: int = 20
_DEFAULT_LR: float = 1.0e-4
_DEFAULT_WEIGHT_DECAY: float = 1.0e-5
_DEFAULT_SCHEDULER: str = "cosine"
_DEFAULT_SEED: int = 42
_DEFAULT_DROPOUT: float = 0.0
_DEFAULT_ATTENTION_HIDDEN_DIM: int = 256
_DEFAULT_CLASSIFIER_HIDDEN_DIM: int = 256
_DEFAULT_GRAD_CLIP_NORM: Optional[float] = None
_DEFAULT_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_DEFAULT_EPS: float = 1.0e-12


class ABMILError(RuntimeError):
    """Base exception for ABMIL baseline failures."""


class ABMILSchemaError(ABMILError):
    """Raised when ABMIL input/output schema contracts are violated."""


class ABMILTrainingError(ABMILError):
    """Raised for invalid or failed ABMIL training flows."""


@dataclass(frozen=True)
class BagSample:
    """Single MIL bag sample.

    Attributes:
        features: Patch feature matrix [N, 768].
        label: Integer class label in [0, num_classes-1].
        slide_id: Optional slide identifier for diagnostics.
        valid_mask: Optional boolean/0-1 vector [N] for valid patches.
    """

    features: np.ndarray
    label: int
    slide_id: str = ""
    valid_mask: Optional[np.ndarray] = None


class _BagDataset(torch.utils.data.Dataset):
    """Internal map-style dataset for ABMIL."""

    def __init__(self, samples: Sequence[BagSample], feature_dim: int = _FEATURE_DIM) -> None:
        if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a sequence of BagSample.")
        if len(samples) <= 0:
            raise ABMILSchemaError("samples cannot be empty.")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be an integer.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0.")

        self.samples: List[BagSample] = list(samples)
        self.feature_dim: int = int(feature_dim)

        for idx, sample in enumerate(self.samples):
            self._validate_sample(sample=sample, index=idx, feature_dim=self.feature_dim)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample: BagSample = self.samples[idx]
        features_np: np.ndarray = np.asarray(sample.features, dtype=np.float32)
        valid_mask_np: Optional[np.ndarray]
        if sample.valid_mask is None:
            valid_mask_np = None
        else:
            valid_mask_np = _to_bool_mask(
                sample.valid_mask,
                expected_n=int(features_np.shape[0]),
                name=f"samples[{idx}].valid_mask",
            )

        return {
            "features": torch.from_numpy(features_np),
            "label": int(sample.label),
            "slide_id": str(sample.slide_id),
            "valid_mask": None if valid_mask_np is None else torch.from_numpy(valid_mask_np),
        }

    @staticmethod
    def _validate_sample(sample: Any, index: int, feature_dim: int) -> None:
        if not isinstance(sample, BagSample):
            raise TypeError(f"samples[{index}] must be BagSample, got {type(sample).__name__}.")

        feats: np.ndarray = np.asarray(sample.features)
        if feats.ndim != 2:
            raise ABMILSchemaError(
                f"samples[{index}].features must be rank-2 [N,D], got {tuple(feats.shape)}."
            )
        if int(feats.shape[0]) <= 0:
            raise ABMILSchemaError(f"samples[{index}].features has zero rows.")
        if int(feats.shape[1]) != int(feature_dim):
            raise ABMILSchemaError(
                f"samples[{index}].features dim must be {feature_dim}, got {int(feats.shape[1])}."
            )
        if not np.isfinite(feats).all():
            raise ABMILSchemaError(f"samples[{index}].features contains NaN/Inf.")

        if isinstance(sample.label, bool) or not isinstance(sample.label, (int, np.integer)):
            raise TypeError(f"samples[{index}].label must be integer.")
        if int(sample.label) < 0:
            raise ABMILSchemaError(f"samples[{index}].label must be >= 0.")

        if sample.valid_mask is not None:
            _to_bool_mask(
                sample.valid_mask,
                expected_n=int(feats.shape[0]),
                name=f"samples[{index}].valid_mask",
            )


def _abmil_collate(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Collate function for variable-length bag batches.

    This implementation is strict and designed for batch size = 1 to match the
    paper protocol. It raises if a different batch size is used.
    """
    if not isinstance(batch, Sequence) or len(batch) <= 0:
        raise ABMILSchemaError("Collate batch must be non-empty.")
    if len(batch) != _DEFAULT_BATCH_SIZE:
        raise ABMILSchemaError(
            f"ABMIL baseline expects batch size {_DEFAULT_BATCH_SIZE}, got {len(batch)}."
        )
    item: Mapping[str, Any] = batch[0]
    return {
        "features": item["features"],
        "label": torch.tensor(int(item["label"]), dtype=torch.long),
        "slide_id": str(item["slide_id"]),
        "valid_mask": item["valid_mask"],
    }


class ABMILModel(nn.Module):
    """Attention-based MIL classifier for patch-feature bags."""

    def __init__(
        self,
        in_dim: int = _FEATURE_DIM,
        num_classes: int = 2,
        attention_hidden_dim: int = _DEFAULT_ATTENTION_HIDDEN_DIM,
        classifier_hidden_dim: int = _DEFAULT_CLASSIFIER_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if isinstance(in_dim, bool) or not isinstance(in_dim, int):
            raise TypeError("in_dim must be integer.")
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be integer.")
        if isinstance(attention_hidden_dim, bool) or not isinstance(attention_hidden_dim, int):
            raise TypeError("attention_hidden_dim must be integer.")
        if isinstance(classifier_hidden_dim, bool) or not isinstance(classifier_hidden_dim, int):
            raise TypeError("classifier_hidden_dim must be integer.")
        if not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be numeric.")

        if in_dim <= 0:
            raise ValueError("in_dim must be > 0.")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1.")
        if attention_hidden_dim <= 0:
            raise ValueError("attention_hidden_dim must be > 0.")
        if classifier_hidden_dim <= 0:
            raise ValueError("classifier_hidden_dim must be > 0.")
        if float(dropout) < 0.0 or float(dropout) >= 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.in_dim: int = int(in_dim)
        self.num_classes: int = int(num_classes)
        self.attention_hidden_dim: int = int(attention_hidden_dim)
        self.classifier_hidden_dim: int = int(classifier_hidden_dim)
        self.dropout_p: float = float(dropout)

        self.pre_attn: nn.Sequential = nn.Sequential(
            nn.Linear(self.in_dim, self.classifier_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.dropout_p),
        )
        self.attn_net: nn.Sequential = nn.Sequential(
            nn.Linear(self.classifier_hidden_dim, self.attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.attention_hidden_dim, 1),
        )
        self.classifier: nn.Sequential = nn.Sequential(
            nn.Linear(self.classifier_hidden_dim, self.classifier_hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(self.classifier_hidden_dim, self.num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Tensor [N, D].
            valid_mask: Optional bool mask [N].

        Returns:
            Dict with:
            - logits: [C]
            - pooled_embedding: [H]
            - attention_weights: [N]
            - valid_indices: [M] selected token indices
        """
        if not isinstance(features, torch.Tensor):
            raise TypeError("features must be torch.Tensor.")
        if features.ndim != 2:
            raise ABMILSchemaError(f"features must be rank-2 [N,D], got {tuple(features.shape)}.")
        if int(features.shape[0]) <= 0:
            raise ABMILSchemaError("features cannot be empty.")
        if int(features.shape[1]) != self.in_dim:
            raise ABMILSchemaError(
                f"features second dimension must be {self.in_dim}, got {int(features.shape[1])}."
            )
        if not torch.isfinite(features).all():
            raise ABMILSchemaError("features contains non-finite values.")

        valid_idx: torch.Tensor = self._resolve_valid_indices(
            valid_mask=valid_mask,
            n_tokens=int(features.shape[0]),
            device=features.device,
        )
        bag: torch.Tensor = features.index_select(dim=0, index=valid_idx)
        if int(bag.shape[0]) <= 0:
            raise ABMILSchemaError("No valid instances remain after masking.")

        bag_hidden: torch.Tensor = self.pre_attn(bag)  # [M, H]
        attn_logits: torch.Tensor = self.attn_net(bag_hidden).squeeze(-1)  # [M]
        attn_weights_valid: torch.Tensor = torch.softmax(attn_logits, dim=0)  # [M]

        pooled: torch.Tensor = torch.sum(bag_hidden * attn_weights_valid.unsqueeze(-1), dim=0)  # [H]
        logits: torch.Tensor = self.classifier(pooled)  # [C]

        full_attn: torch.Tensor = torch.zeros(
            (int(features.shape[0]),),
            dtype=attn_weights_valid.dtype,
            device=attn_weights_valid.device,
        )
        full_attn.scatter_(0, valid_idx, attn_weights_valid)

        return {
            "logits": logits,
            "pooled_embedding": pooled,
            "attention_weights": full_attn,
            "valid_indices": valid_idx,
        }

    @staticmethod
    def _resolve_valid_indices(
        valid_mask: Optional[torch.Tensor],
        n_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        if valid_mask is None:
            return torch.arange(n_tokens, dtype=torch.long, device=device)

        if not isinstance(valid_mask, torch.Tensor):
            raise TypeError("valid_mask must be torch.Tensor or None.")
        if valid_mask.ndim != 1:
            raise ABMILSchemaError(
                f"valid_mask must be rank-1 [N], got {tuple(valid_mask.shape)}."
            )
        if int(valid_mask.shape[0]) != n_tokens:
            raise ABMILSchemaError(
                f"valid_mask length mismatch: {int(valid_mask.shape[0])} vs {n_tokens}."
            )
        mask_bool: torch.Tensor = valid_mask.to(device=device, dtype=torch.bool)
        valid_idx: torch.Tensor = torch.nonzero(mask_bool, as_tuple=False).squeeze(1).to(torch.long)
        if valid_idx.numel() <= 0:
            raise ABMILSchemaError("valid_mask selects zero instances.")
        return valid_idx


class ABMILBaseline:
    """Train/inference wrapper for ABMIL supervised baseline."""

    def __init__(
        self,
        num_classes: int,
        feature_dim: int = _FEATURE_DIM,
        patch_size_px: int = _PATCH_SIZE_PX,
        magnification: str = _MAGNIFICATION,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        epochs: int = _DEFAULT_EPOCHS,
        lr: float = _DEFAULT_LR,
        weight_decay: float = _DEFAULT_WEIGHT_DECAY,
        scheduler: str = _DEFAULT_SCHEDULER,
        attention_hidden_dim: int = _DEFAULT_ATTENTION_HIDDEN_DIM,
        classifier_hidden_dim: int = _DEFAULT_CLASSIFIER_HIDDEN_DIM,
        dropout: float = _DEFAULT_DROPOUT,
        grad_clip_norm: Optional[float] = _DEFAULT_GRAD_CLIP_NORM,
        seed: int = _DEFAULT_SEED,
        device: str = _DEFAULT_DEVICE,
    ) -> None:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be integer.")
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1.")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise TypeError("feature_dim must be integer.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be > 0.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if isinstance(epochs, bool) or not isinstance(epochs, int):
            raise TypeError("epochs must be integer.")
        if epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be numeric.")
        if float(lr) <= 0.0:
            raise ValueError("lr must be > 0.")
        if not isinstance(weight_decay, (int, float)):
            raise TypeError("weight_decay must be numeric.")
        if float(weight_decay) < 0.0:
            raise ValueError("weight_decay must be >= 0.")
        if not isinstance(scheduler, str) or not scheduler.strip():
            raise TypeError("scheduler must be non-empty string.")
        if grad_clip_norm is not None and not isinstance(grad_clip_norm, (int, float)):
            raise TypeError("grad_clip_norm must be numeric or None.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be integer.")
        if seed < 0:
            raise ValueError("seed must be >= 0.")
        if not isinstance(device, str) or not device.strip():
            raise TypeError("device must be non-empty string.")

        self.num_classes: int = int(num_classes)
        self.feature_dim: int = int(feature_dim)
        self.patch_size_px: int = int(patch_size_px)
        self.magnification: str = str(magnification)
        self.batch_size: int = int(batch_size)
        self.epochs: int = int(epochs)
        self.lr: float = float(lr)
        self.weight_decay: float = float(weight_decay)
        self.scheduler_name: str = str(scheduler).strip().lower()
        self.attention_hidden_dim: int = int(attention_hidden_dim)
        self.classifier_hidden_dim: int = int(classifier_hidden_dim)
        self.dropout: float = float(dropout)
        self.grad_clip_norm: Optional[float] = (
            None if grad_clip_norm is None else float(grad_clip_norm)
        )
        self.seed: int = int(seed)
        self.device: torch.device = self._resolve_device(device)
        self.eps: float = _DEFAULT_EPS

        self.stage1_region_grid: Tuple[int, int] = _STAGE1_REGION_GRID
        self.stage3_crop_grid: Tuple[int, int] = _STAGE3_CROP_GRID

        self._validate_config_invariants()
        self._seed_everything()

        self.model: ABMILModel = ABMILModel(
            in_dim=self.feature_dim,
            num_classes=self.num_classes,
            attention_hidden_dim=self.attention_hidden_dim,
            classifier_hidden_dim=self.classifier_hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        self._optimizer: AdamW = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self._scheduler: Optional[CosineAnnealingLR] = self._build_scheduler()
        self._criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
        self._is_fitted: bool = False

    def fit(
        self,
        train_samples: Sequence[BagSample],
        val_samples: Optional[Sequence[BagSample]] = None,
    ) -> Dict[str, Any]:
        """Train ABMIL on supervised bags.

        Args:
            train_samples: Training bag sequence.
            val_samples: Optional validation sequence. If provided, best model is
                selected by minimum validation loss.

        Returns:
            Training summary dictionary with epoch history and best epoch.
        """
        train_ds: _BagDataset = _BagDataset(samples=train_samples, feature_dim=self.feature_dim)
        val_ds: Optional[_BagDataset] = (
            None if val_samples is None else _BagDataset(samples=val_samples, feature_dim=self.feature_dim)
        )
        self._validate_label_range(dataset=train_ds, name="train_samples")
        if val_ds is not None:
            self._validate_label_range(dataset=val_ds, name="val_samples")

        train_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=bool(self.device.type == "cuda"),
            collate_fn=_abmil_collate,
            drop_last=False,
        )
        val_loader: Optional[torch.utils.data.DataLoader] = None
        if val_ds is not None:
            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=bool(self.device.type == "cuda"),
                collate_fn=_abmil_collate,
                drop_last=False,
            )

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_val_loss: float = float("inf")
        best_epoch: int = -1
        history: List[Dict[str, float]] = []

        for epoch_idx in range(self.epochs):
            train_stats: Dict[str, float] = self._run_epoch_train(train_loader=train_loader)
            val_stats: Dict[str, float] = {}
            if val_loader is not None:
                val_stats = self._run_epoch_eval(loader=val_loader)
                current_val_loss: float = float(val_stats["loss"])
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = int(epoch_idx)
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }

            epoch_record: Dict[str, float] = {
                "epoch": float(epoch_idx),
                "train_loss": float(train_stats["loss"]),
                "train_balanced_accuracy": float(train_stats["balanced_accuracy"]),
                "train_weighted_f1": float(train_stats["weighted_f1"]),
            }
            if val_stats:
                epoch_record.update(
                    {
                        "val_loss": float(val_stats["loss"]),
                        "val_balanced_accuracy": float(val_stats["balanced_accuracy"]),
                        "val_weighted_f1": float(val_stats["weighted_f1"]),
                    }
                )
                if "auroc" in val_stats:
                    epoch_record["val_auroc"] = float(val_stats["auroc"])
            history.append(epoch_record)

        if val_loader is not None and best_state is not None:
            self.model.load_state_dict(best_state, strict=True)

        self._is_fitted = True
        result: Dict[str, Any] = {
            "task": "abmil_supervised_training",
            "protocol": {
                "batch_size": int(self.batch_size),
                "epochs": int(self.epochs),
                "optimizer": "adamw",
                "learning_rate": float(self.lr),
                "weight_decay": float(self.weight_decay),
                "scheduler": str(self.scheduler_name),
                "validation_checkpoint_selection": bool(val_loader is not None),
            },
            "best_epoch": int(best_epoch),
            "best_val_loss": None if best_epoch < 0 else float(best_val_loss),
            "history": history,
        }
        return result

    def predict_logits(self, samples: Sequence[BagSample]) -> np.ndarray:
        """Predict logits [N, C] for bag samples."""
        outputs: Dict[str, np.ndarray] = self._predict_common(samples=samples)
        return outputs["logits"]

    def predict_proba(self, samples: Sequence[BagSample]) -> np.ndarray:
        """Predict probabilities [N, C] for bag samples."""
        outputs: Dict[str, np.ndarray] = self._predict_common(samples=samples)
        return outputs["proba"]

    def predict(self, samples: Sequence[BagSample]) -> np.ndarray:
        """Predict class indices [N] for bag samples."""
        outputs: Dict[str, np.ndarray] = self._predict_common(samples=samples)
        return outputs["y_pred"]

    def embed_slides(self, samples: Sequence[BagSample]) -> np.ndarray:
        """Return pooled slide embeddings [N, H] from trained ABMIL."""
        outputs: Dict[str, np.ndarray] = self._predict_common(samples=samples)
        return outputs["embeddings"]

    def evaluate(self, samples: Sequence[BagSample]) -> Dict[str, Any]:
        """Evaluate trained ABMIL on supervised samples."""
        dataset: _BagDataset = _BagDataset(samples=samples, feature_dim=self.feature_dim)
        self._validate_label_range(dataset=dataset, name="samples")
        outputs: Dict[str, np.ndarray] = self._predict_common(samples=samples)

        y_true: np.ndarray = np.asarray([sample.label for sample in samples], dtype=np.int64)
        y_pred: np.ndarray = outputs["y_pred"]
        proba: np.ndarray = outputs["proba"]

        metrics: Dict[str, float] = {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        }
        if self.num_classes == 2:
            metrics["auroc"] = self._safe_binary_auroc(y_true=y_true, y_score_binary=proba[:, 1])

        return {
            "task": "abmil_supervised_evaluation",
            "num_samples": int(len(samples)),
            "num_classes": int(self.num_classes),
            "metrics": metrics,
            "outputs": {
                "y_true": y_true.astype(np.int64, copy=False).tolist(),
                "y_pred": y_pred.astype(np.int64, copy=False).tolist(),
            },
        }

    def save(self, path: str) -> None:
        """Save model checkpoint."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be non-empty string.")
        payload: Dict[str, Any] = {
            "state_dict": self.model.state_dict(),
            "num_classes": int(self.num_classes),
            "feature_dim": int(self.feature_dim),
            "patch_size_px": int(self.patch_size_px),
            "magnification": str(self.magnification),
            "attention_hidden_dim": int(self.attention_hidden_dim),
            "classifier_hidden_dim": int(self.classifier_hidden_dim),
            "dropout": float(self.dropout),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: str = _DEFAULT_DEVICE) -> "ABMILBaseline":
        """Load ABMIL baseline from checkpoint."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be non-empty string.")
        checkpoint: Any = torch.load(path, map_location="cpu")
        if not isinstance(checkpoint, Mapping):
            raise ABMILSchemaError("Checkpoint payload must be a mapping.")
        required_keys: Tuple[str, ...] = (
            "state_dict",
            "num_classes",
            "feature_dim",
            "patch_size_px",
            "magnification",
            "attention_hidden_dim",
            "classifier_hidden_dim",
            "dropout",
        )
        for key in required_keys:
            if key not in checkpoint:
                raise ABMILSchemaError(f"Checkpoint missing key: {key}.")

        baseline: ABMILBaseline = cls(
            num_classes=int(checkpoint["num_classes"]),
            feature_dim=int(checkpoint["feature_dim"]),
            patch_size_px=int(checkpoint["patch_size_px"]),
            magnification=str(checkpoint["magnification"]),
            attention_hidden_dim=int(checkpoint["attention_hidden_dim"]),
            classifier_hidden_dim=int(checkpoint["classifier_hidden_dim"]),
            dropout=float(checkpoint["dropout"]),
            device=device,
        )
        state_dict_obj: Any = checkpoint["state_dict"]
        if not isinstance(state_dict_obj, Mapping):
            raise ABMILSchemaError("Checkpoint state_dict must be a mapping.")
        baseline.model.load_state_dict(state_dict_obj, strict=True)
        baseline._is_fitted = True
        return baseline

    def _predict_common(self, samples: Sequence[BagSample]) -> Dict[str, np.ndarray]:
        if not self._is_fitted:
            raise ABMILTrainingError("Model is not fitted. Call fit() or load() first.")
        dataset: _BagDataset = _BagDataset(samples=samples, feature_dim=self.feature_dim)
        self._validate_label_range(dataset=dataset, name="samples")

        loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=bool(self.device.type == "cuda"),
            collate_fn=_abmil_collate,
            drop_last=False,
        )

        self.model.eval()
        logits_all: List[np.ndarray] = []
        proba_all: List[np.ndarray] = []
        y_pred_all: List[int] = []
        emb_all: List[np.ndarray] = []
        attn_all: List[np.ndarray] = []

        with torch.inference_mode():
            for batch in loader:
                features: torch.Tensor = batch["features"].to(self.device, dtype=torch.float32)
                valid_mask: Optional[torch.Tensor]
                if batch["valid_mask"] is None:
                    valid_mask = None
                else:
                    valid_mask = batch["valid_mask"].to(self.device, dtype=torch.bool)

                out: Dict[str, torch.Tensor] = self.model(features=features, valid_mask=valid_mask)
                logits_t: torch.Tensor = out["logits"].reshape(1, -1)
                proba_t: torch.Tensor = torch.softmax(logits_t, dim=1)
                y_pred_t: torch.Tensor = torch.argmax(proba_t, dim=1)

                logits_np: np.ndarray = logits_t.detach().cpu().numpy().astype(np.float32, copy=False)
                proba_np: np.ndarray = proba_t.detach().cpu().numpy().astype(np.float32, copy=False)
                y_pred_np: np.ndarray = y_pred_t.detach().cpu().numpy().astype(np.int64, copy=False)
                emb_np: np.ndarray = (
                    out["pooled_embedding"].detach().cpu().numpy().astype(np.float32, copy=False)
                )
                attn_np: np.ndarray = (
                    out["attention_weights"].detach().cpu().numpy().astype(np.float32, copy=False)
                )

                logits_all.append(logits_np)
                proba_all.append(proba_np)
                y_pred_all.append(int(y_pred_np[0]))
                emb_all.append(emb_np)
                attn_all.append(attn_np)

        logits: np.ndarray = np.concatenate(logits_all, axis=0)
        proba: np.ndarray = np.concatenate(proba_all, axis=0)
        y_pred: np.ndarray = np.asarray(y_pred_all, dtype=np.int64)
        embeddings: np.ndarray = np.stack(emb_all, axis=0)

        if logits.shape != (len(samples), self.num_classes):
            raise ABMILSchemaError(
                f"Logits shape mismatch: expected {(len(samples), self.num_classes)}, got {logits.shape}."
            )
        if proba.shape != (len(samples), self.num_classes):
            raise ABMILSchemaError(
                f"Proba shape mismatch: expected {(len(samples), self.num_classes)}, got {proba.shape}."
            )
        if embeddings.shape[0] != len(samples):
            raise ABMILSchemaError(
                f"Embedding row mismatch: expected {len(samples)}, got {embeddings.shape[0]}."
            )
        if not np.isfinite(logits).all() or not np.isfinite(proba).all() or not np.isfinite(embeddings).all():
            raise ABMILSchemaError("Non-finite outputs produced by ABMIL inference.")

        return {
            "logits": logits,
            "proba": proba,
            "y_pred": y_pred,
            "embeddings": embeddings,
            "attention_weights": np.asarray(attn_all, dtype=object),
        }

    def _run_epoch_train(self, train_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.train()
        loss_sum: float = 0.0
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score_binary: List[float] = []

        for batch in train_loader:
            self._optimizer.zero_grad(set_to_none=True)
            features: torch.Tensor = batch["features"].to(self.device, dtype=torch.float32)
            label: torch.Tensor = batch["label"].to(self.device, dtype=torch.long).reshape(1)
            valid_mask: Optional[torch.Tensor]
            if batch["valid_mask"] is None:
                valid_mask = None
            else:
                valid_mask = batch["valid_mask"].to(self.device, dtype=torch.bool)

            out: Dict[str, torch.Tensor] = self.model(features=features, valid_mask=valid_mask)
            logits_1: torch.Tensor = out["logits"].reshape(1, -1)
            if int(logits_1.shape[1]) != self.num_classes:
                raise ABMILSchemaError(
                    f"Model logits class dim mismatch: expected {self.num_classes}, got {int(logits_1.shape[1])}."
                )
            loss: torch.Tensor = self._criterion(logits_1, label)
            if not torch.isfinite(loss):
                raise ABMILTrainingError("Non-finite loss encountered during training.")

            loss.backward()
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(self.grad_clip_norm))
            self._optimizer.step()

            loss_sum += float(loss.detach().cpu().item())
            proba: torch.Tensor = torch.softmax(logits_1, dim=1)
            pred: torch.Tensor = torch.argmax(proba, dim=1)
            y_true.append(int(label.item()))
            y_pred.append(int(pred.item()))
            if self.num_classes == 2:
                y_score_binary.append(float(proba[:, 1].detach().cpu().item()))

        if self._scheduler is not None:
            self._scheduler.step()

        return self._build_metrics(
            loss_sum=loss_sum,
            n_samples=len(train_loader),
            y_true=np.asarray(y_true, dtype=np.int64),
            y_pred=np.asarray(y_pred, dtype=np.int64),
            y_score_binary=None
            if self.num_classes != 2
            else np.asarray(y_score_binary, dtype=np.float64),
        )

    def _run_epoch_eval(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        loss_sum: float = 0.0
        y_true: List[int] = []
        y_pred: List[int] = []
        y_score_binary: List[float] = []

        with torch.inference_mode():
            for batch in loader:
                features: torch.Tensor = batch["features"].to(self.device, dtype=torch.float32)
                label: torch.Tensor = batch["label"].to(self.device, dtype=torch.long).reshape(1)
                valid_mask: Optional[torch.Tensor]
                if batch["valid_mask"] is None:
                    valid_mask = None
                else:
                    valid_mask = batch["valid_mask"].to(self.device, dtype=torch.bool)

                out: Dict[str, torch.Tensor] = self.model(features=features, valid_mask=valid_mask)
                logits_1: torch.Tensor = out["logits"].reshape(1, -1)
                loss: torch.Tensor = self._criterion(logits_1, label)
                if not torch.isfinite(loss):
                    raise ABMILTrainingError("Non-finite loss encountered during evaluation.")

                loss_sum += float(loss.detach().cpu().item())
                proba: torch.Tensor = torch.softmax(logits_1, dim=1)
                pred: torch.Tensor = torch.argmax(proba, dim=1)
                y_true.append(int(label.item()))
                y_pred.append(int(pred.item()))
                if self.num_classes == 2:
                    y_score_binary.append(float(proba[:, 1].detach().cpu().item()))

        return self._build_metrics(
            loss_sum=loss_sum,
            n_samples=len(loader),
            y_true=np.asarray(y_true, dtype=np.int64),
            y_pred=np.asarray(y_pred, dtype=np.int64),
            y_score_binary=None
            if self.num_classes != 2
            else np.asarray(y_score_binary, dtype=np.float64),
        )

    def _build_metrics(
        self,
        loss_sum: float,
        n_samples: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score_binary: Optional[np.ndarray],
    ) -> Dict[str, float]:
        if n_samples <= 0:
            raise ABMILSchemaError("n_samples must be > 0.")
        if y_true.shape != y_pred.shape:
            raise ABMILSchemaError("y_true and y_pred must have matching shapes.")
        if y_true.ndim != 1:
            raise ABMILSchemaError("y_true/y_pred must be rank-1.")

        metrics: Dict[str, float] = {
            "loss": float(loss_sum / max(float(n_samples), self.eps)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        }
        if self.num_classes == 2 and y_score_binary is not None:
            metrics["auroc"] = self._safe_binary_auroc(
                y_true=y_true,
                y_score_binary=y_score_binary,
            )
        return metrics

    def _build_scheduler(self) -> Optional[CosineAnnealingLR]:
        if self.scheduler_name in {"none", "off", "disabled"}:
            return None
        if self.scheduler_name != _DEFAULT_SCHEDULER:
            raise ValueError(
                f"Unsupported scheduler '{self.scheduler_name}'. Only '{_DEFAULT_SCHEDULER}' is supported."
            )
        return CosineAnnealingLR(self._optimizer, T_max=max(1, self.epochs))

    @staticmethod
    def _safe_binary_auroc(y_true: np.ndarray, y_score_binary: np.ndarray) -> float:
        if y_true.ndim != 1 or y_score_binary.ndim != 1:
            raise ABMILSchemaError("y_true and y_score_binary must be rank-1 arrays.")
        if int(y_true.shape[0]) != int(y_score_binary.shape[0]):
            raise ABMILSchemaError(
                "y_true and y_score_binary must have same length. "
                f"Got {int(y_true.shape[0])} and {int(y_score_binary.shape[0])}."
            )
        if int(np.unique(y_true).shape[0]) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score_binary))

    def _validate_label_range(self, dataset: _BagDataset, name: str) -> None:
        min_label: int = min(int(sample.label) for sample in dataset.samples)
        max_label: int = max(int(sample.label) for sample in dataset.samples)
        if min_label < 0:
            raise ABMILSchemaError(f"{name} labels must be >=0, found {min_label}.")
        if max_label >= self.num_classes:
            raise ABMILSchemaError(
                f"{name} labels out of range for num_classes={self.num_classes}: max={max_label}."
            )

    def _validate_config_invariants(self) -> None:
        if self.feature_dim != _FEATURE_DIM:
            raise ValueError(f"feature_dim must be {_FEATURE_DIM}, got {self.feature_dim}.")
        if self.patch_size_px != _PATCH_SIZE_PX:
            raise ValueError(f"patch_size_px must be {_PATCH_SIZE_PX}, got {self.patch_size_px}.")
        if self.magnification != _MAGNIFICATION:
            raise ValueError(f"magnification must be '{_MAGNIFICATION}', got '{self.magnification}'.")
        if self.batch_size != _DEFAULT_BATCH_SIZE:
            raise ValueError(
                f"ABMIL protocol requires batch_size={_DEFAULT_BATCH_SIZE}, got {self.batch_size}."
            )
        if self.epochs != _DEFAULT_EPOCHS:
            raise ValueError(
                f"ABMIL protocol requires epochs={_DEFAULT_EPOCHS}, got {self.epochs}."
            )
        if not np.isclose(self.lr, _DEFAULT_LR, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"ABMIL protocol requires lr={_DEFAULT_LR}, got {self.lr}.")
        if not np.isclose(self.weight_decay, _DEFAULT_WEIGHT_DECAY, rtol=0.0, atol=1.0e-12):
            raise ValueError(
                f"ABMIL protocol requires weight_decay={_DEFAULT_WEIGHT_DECAY}, got {self.weight_decay}."
            )
        if self.scheduler_name != _DEFAULT_SCHEDULER:
            raise ValueError(
                f"ABMIL protocol requires scheduler='{_DEFAULT_SCHEDULER}', got '{self.scheduler_name}'."
            )

    def _seed_everything(self) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        normalized: str = str(device).strip().lower()
        if normalized.startswith("cuda") and torch.cuda.is_available():
            return torch.device(normalized)
        if normalized == "cpu":
            return torch.device("cpu")
        if normalized.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(_DEFAULT_DEVICE)


class Baseline(ABMILBaseline):
    """Design-compat alias."""


def run_abmil(
    train_samples: Sequence[BagSample],
    test_samples: Sequence[BagSample],
    val_samples: Optional[Sequence[BagSample]] = None,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """Convenience API: fit ABMIL and evaluate on test samples.

    Args:
        train_samples: Supervised train bags.
        test_samples: Supervised test bags.
        val_samples: Optional validation bags.
        num_classes: Optional class count. If None, inferred from train labels.

    Returns:
        Dictionary with training summary and test metrics.
    """
    if not isinstance(train_samples, Sequence) or len(train_samples) <= 0:
        raise ABMILSchemaError("train_samples must be a non-empty sequence.")
    if not isinstance(test_samples, Sequence) or len(test_samples) <= 0:
        raise ABMILSchemaError("test_samples must be a non-empty sequence.")

    inferred_num_classes: int
    if num_classes is None:
        label_values: List[int] = [int(sample.label) for sample in train_samples]
        inferred_num_classes = int(max(label_values) + 1)
    else:
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes must be integer when provided.")
        inferred_num_classes = int(num_classes)

    baseline: ABMILBaseline = ABMILBaseline(num_classes=inferred_num_classes)
    train_summary: Dict[str, Any] = baseline.fit(train_samples=train_samples, val_samples=val_samples)
    test_result: Dict[str, Any] = baseline.evaluate(samples=test_samples)
    return {
        "train": train_summary,
        "test": test_result,
    }


def _to_bool_mask(mask: Any, expected_n: int, name: str) -> np.ndarray:
    arr: np.ndarray = np.asarray(mask)
    if arr.ndim != 1:
        raise ABMILSchemaError(f"{name} must be rank-1 [N], got {tuple(arr.shape)}.")
    if int(arr.shape[0]) != int(expected_n):
        raise ABMILSchemaError(
            f"{name} length mismatch: expected {expected_n}, got {int(arr.shape[0])}."
        )

    if np.issubdtype(arr.dtype, np.bool_):
        out: np.ndarray = arr.astype(np.bool_, copy=False)
    elif np.issubdtype(arr.dtype, np.integer):
        unique_values: np.ndarray = np.unique(arr.astype(np.int64, copy=False))
        if not set(int(v) for v in unique_values.tolist()).issubset({0, 1}):
            raise ABMILSchemaError(f"{name} integer mask must be binary 0/1.")
        out = arr.astype(np.bool_, copy=False)
    elif np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.isfinite(arr)):
            raise ABMILSchemaError(f"{name} contains NaN/Inf.")
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ABMILSchemaError(f"{name} float mask must be integer-valued.")
        unique_values = np.unique(arr.astype(np.int64, copy=False))
        if not set(int(v) for v in unique_values.tolist()).issubset({0, 1}):
            raise ABMILSchemaError(f"{name} float mask must be binary 0/1.")
        out = arr.astype(np.bool_, copy=False)
    else:
        raise ABMILSchemaError(f"{name} has unsupported dtype={arr.dtype}.")

    if int(np.sum(out.astype(np.int64))) <= 0:
        raise ABMILSchemaError(f"{name} selects zero instances.")
    return out


__all__ = [
    "ABMILError",
    "ABMILSchemaError",
    "ABMILTrainingError",
    "BagSample",
    "ABMILModel",
    "ABMILBaseline",
    "Baseline",
    "run_abmil",
]
