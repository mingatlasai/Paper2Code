"""Contrastive losses for THREADS multimodal pretraining.

This module implements the design-locked contract:
- ``ContrastiveLoss.__init__(temperature: float, bidirectional: bool) -> None``
- ``ContrastiveLoss.forward(z_wsi: object, z_mol: object) -> float``

Paper/config alignment:
- Objective family: InfoNCE-style cross-modal contrastive learning.
- Shared embedding space width: 1024 (validated upstream by encoders).
- Bidirectional contrastive term is supported and enabled by default.

Notes:
- This class is modality-agnostic: ``z_mol`` may come from RNA or DNA branches.
- The implementation uses numerically stable cross-entropy over similarity logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as torch_functional


# ----------------------------------------------------------------------------
# Defaults and constants
# ----------------------------------------------------------------------------
# Temperature is intentionally not hardcoded to a paper-specific numeric value
# because the provided configuration marks it unresolved (null). A NaN sentinel
# forces explicit runtime configuration while still providing a signature default.
DEFAULT_TEMPERATURE: float = float("nan")
DEFAULT_BIDIRECTIONAL: bool = True

DEFAULT_EPS: float = 1.0e-12
DEFAULT_LABEL_DTYPE: torch.dtype = torch.long


class ContrastiveLossError(Exception):
    """Base exception for contrastive loss failures."""


class ContrastiveLossConfigError(ContrastiveLossError):
    """Raised when constructor configuration is invalid."""


class ContrastiveLossInputError(ContrastiveLossError):
    """Raised when forward inputs violate required contracts."""


@dataclass(frozen=True)
class _BatchShape:
    """Validated input shape metadata for one forward call."""

    batch_size: int
    embedding_dim: int


class ContrastiveLoss(nn.Module):
    """InfoNCE-style contrastive loss with optional bidirectional averaging.

    Forward semantics:
    1. L2-normalize ``z_wsi`` and ``z_mol`` over embedding dimension.
    2. Compute similarity logits ``S = (z_wsi @ z_mol.T) / temperature``.
    3. Compute cross-entropy with diagonal positives.
    4. If ``bidirectional=True``, average WSI->MOL and MOL->WSI losses.
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        bidirectional: bool = DEFAULT_BIDIRECTIONAL,
    ) -> None:
        """Initialize contrastive loss module.

        Args:
            temperature: Positive scalar temperature for InfoNCE logits.
            bidirectional: Whether to include both WSI->MOL and MOL->WSI terms.

        Raises:
            ContrastiveLossConfigError: If arguments are invalid.
        """
        super().__init__()

        validated_temperature: float = self._validate_temperature(temperature)
        validated_bidirectional: bool = self._validate_bidirectional(bidirectional)

        self._temperature: float = validated_temperature
        self._bidirectional: bool = validated_bidirectional

    @property
    def temperature(self) -> float:
        """Return configured temperature."""
        return self._temperature

    @property
    def bidirectional(self) -> bool:
        """Return bidirectional flag."""
        return self._bidirectional

    def forward(self, z_wsi: object, z_mol: object) -> torch.Tensor:
        """Compute contrastive loss from paired embedding batches.

        Args:
            z_wsi: Tensor-like with shape ``[B, D]``.
            z_mol: Tensor-like with shape ``[B, D]``.

        Returns:
            Scalar tensor loss.

        Raises:
            ContrastiveLossInputError: If inputs are malformed.
        """
        wsi_tensor: torch.Tensor = self._coerce_embedding_tensor(z_wsi, "z_wsi")
        mol_tensor: torch.Tensor = self._coerce_embedding_tensor(z_mol, "z_mol")
        shape_info: _BatchShape = self._validate_shapes(wsi_tensor, mol_tensor)

        # Keep matmul/softmax numerics stable under mixed precision by promoting
        # to float32 for similarity/logit computation.
        wsi_fp32: torch.Tensor = wsi_tensor.to(torch.float32)
        mol_fp32: torch.Tensor = mol_tensor.to(torch.float32)

        wsi_normalized: torch.Tensor = torch_functional.normalize(
            wsi_fp32,
            p=2.0,
            dim=1,
            eps=DEFAULT_EPS,
        )
        mol_normalized: torch.Tensor = torch_functional.normalize(
            mol_fp32,
            p=2.0,
            dim=1,
            eps=DEFAULT_EPS,
        )

        logits: torch.Tensor = torch.matmul(wsi_normalized, mol_normalized.transpose(0, 1))
        logits = logits / self._temperature

        if not torch.isfinite(logits).all():
            raise ContrastiveLossInputError("Contrastive logits contain NaN/Inf values.")

        targets: torch.Tensor = torch.arange(
            shape_info.batch_size,
            dtype=DEFAULT_LABEL_DTYPE,
            device=logits.device,
        )

        loss_wsi_to_mol: torch.Tensor = torch_functional.cross_entropy(logits, targets)

        if self._bidirectional:
            loss_mol_to_wsi: torch.Tensor = torch_functional.cross_entropy(logits.transpose(0, 1), targets)
            total_loss: torch.Tensor = 0.5 * (loss_wsi_to_mol + loss_mol_to_wsi)
        else:
            total_loss = loss_wsi_to_mol

        if total_loss.ndim != 0:
            total_loss = total_loss.mean()

        if not torch.isfinite(total_loss):
            raise ContrastiveLossInputError("Contrastive loss is non-finite.")

        return total_loss

    @staticmethod
    def _validate_temperature(temperature: float) -> float:
        if isinstance(temperature, bool):
            raise ContrastiveLossConfigError("temperature must be float, got bool.")

        try:
            value: float = float(temperature)
        except Exception as exc:  # noqa: BLE001
            raise ContrastiveLossConfigError(
                f"temperature must be float, got {temperature!r}."
            ) from exc

        if not torch.isfinite(torch.tensor(value)):
            raise ContrastiveLossConfigError(
                "temperature must be finite and explicitly configured; "
                "got non-finite sentinel/value."
            )
        if value <= 0.0:
            raise ContrastiveLossConfigError(
                f"temperature must be > 0, got {value}."
            )
        return value

    @staticmethod
    def _validate_bidirectional(bidirectional: bool) -> bool:
        if not isinstance(bidirectional, bool):
            raise ContrastiveLossConfigError(
                f"bidirectional must be bool, got {type(bidirectional).__name__}."
            )
        return bool(bidirectional)

    @staticmethod
    def _coerce_embedding_tensor(value: object, field_name: str) -> torch.Tensor:
        try:
            tensor: torch.Tensor = torch.as_tensor(value)
        except Exception as exc:  # noqa: BLE001
            raise ContrastiveLossInputError(
                f"{field_name} cannot be converted to tensor: {exc}"
            ) from exc

        if tensor.ndim != 2:
            raise ContrastiveLossInputError(
                f"{field_name} must have shape [B,D], got {tuple(tensor.shape)}."
            )

        if tensor.shape[0] <= 0:
            raise ContrastiveLossInputError(f"{field_name} batch size must be > 0.")
        if tensor.shape[1] <= 0:
            raise ContrastiveLossInputError(f"{field_name} embedding dim must be > 0.")

        if not torch.is_floating_point(tensor):
            tensor = tensor.to(torch.float32)

        if not torch.isfinite(tensor).all():
            raise ContrastiveLossInputError(
                f"{field_name} contains NaN/Inf values."
            )

        return tensor

    @staticmethod
    def _validate_shapes(z_wsi: torch.Tensor, z_mol: torch.Tensor) -> _BatchShape:
        batch_wsi: int = int(z_wsi.shape[0])
        batch_mol: int = int(z_mol.shape[0])
        emb_wsi: int = int(z_wsi.shape[1])
        emb_mol: int = int(z_mol.shape[1])

        if batch_wsi != batch_mol:
            raise ContrastiveLossInputError(
                "Batch size mismatch between z_wsi and z_mol: "
                f"{batch_wsi} vs {batch_mol}."
            )
        if emb_wsi != emb_mol:
            raise ContrastiveLossInputError(
                "Embedding dimension mismatch between z_wsi and z_mol: "
                f"{emb_wsi} vs {emb_mol}."
            )

        return _BatchShape(batch_size=batch_wsi, embedding_dim=emb_wsi)


__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_BIDIRECTIONAL",
    "ContrastiveLossError",
    "ContrastiveLossConfigError",
    "ContrastiveLossInputError",
    "ContrastiveLoss",
]
