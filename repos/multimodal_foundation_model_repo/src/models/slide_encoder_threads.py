"""THREADS slide encoder implementation (gated ABMIL, single/multi-head).

This module implements the design-locked interface:
- ``ThreadsSlideEncoder.__init__(in_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float)``
- ``ThreadsSlideEncoder.forward(patch_features: object, patch_mask: object) -> object``
- ``ThreadsSlideEncoder.attention_weights() -> object``

Paper/config-aligned defaults:
- Slide encoder type: ABMIL gated attention
- Main attention heads: 2
- Output embedding dim: 1024
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
from torch import nn


DEFAULT_INPUT_DIM: int = 768
DEFAULT_HIDDEN_DIM: int = 1024
DEFAULT_OUTPUT_DIM: int = 1024
DEFAULT_NUM_HEADS: int = 2
DEFAULT_DROPOUT: float = 0.1
DEFAULT_ATTN_DROPOUT: float = 0.25
DEFAULT_EPS: float = 1.0e-12


class SlideEncoderError(Exception):
    """Base exception for slide encoder failures."""


class SlideEncoderConfigError(SlideEncoderError):
    """Raised when encoder configuration is invalid."""


class SlideEncoderInputError(SlideEncoderError):
    """Raised when forward input shape/type is invalid."""


class SlideEncoderRuntimeError(SlideEncoderError):
    """Raised when runtime attention aggregation fails."""


@dataclass(frozen=True)
class _ShapeInfo:
    """Validated shape metadata for one forward call."""

    batch_size: int
    num_patches: int
    feature_dim: int


class _GatedAttentionHead(nn.Module):
    """One gated attention head used by THREADS slide encoder.

    Input shape:
    - features: [B, N, D]
    - mask: [B, N] (True for valid patches)

    Output shape:
    - pooled: [B, D]
    - attention: [B, N]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = DEFAULT_ATTN_DROPOUT,
        eps: float = DEFAULT_EPS,
    ) -> None:
        super().__init__()

        if hidden_dim <= 0:
            raise SlideEncoderConfigError(f"hidden_dim must be > 0, got {hidden_dim}.")
        if not (0.0 <= float(dropout) < 1.0):
            raise SlideEncoderConfigError(
                f"dropout must be in [0, 1), got {dropout}."
            )
        if eps <= 0.0:
            raise SlideEncoderConfigError(f"eps must be > 0, got {eps}.")

        self._hidden_dim: int = int(hidden_dim)
        self._eps: float = float(eps)

        self._a: nn.Linear = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._b: nn.Linear = nn.Linear(self._hidden_dim, self._hidden_dim)
        self._c: nn.Linear = nn.Linear(self._hidden_dim, 1)

        self._dropout: nn.Dropout = nn.Dropout(float(dropout))
        self._tanh: nn.Tanh = nn.Tanh()
        self._sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute masked gated-attention pooling."""
        if features.ndim != 3:
            raise SlideEncoderInputError(
                f"features must have shape [B,N,D], got {tuple(features.shape)}."
            )
        if mask.ndim != 2:
            raise SlideEncoderInputError(
                f"mask must have shape [B,N], got {tuple(mask.shape)}."
            )

        batch_size: int = int(features.shape[0])
        num_patches: int = int(features.shape[1])
        hidden_dim: int = int(features.shape[2])

        if mask.shape[0] != batch_size or mask.shape[1] != num_patches:
            raise SlideEncoderInputError(
                "mask shape mismatch with features: "
                f"features={tuple(features.shape)}, mask={tuple(mask.shape)}."
            )
        if hidden_dim != self._hidden_dim:
            raise SlideEncoderInputError(
                f"feature hidden dim mismatch: expected {self._hidden_dim}, got {hidden_dim}."
            )

        # Gated attention:
        # alpha ~ softmax( W_c( tanh(W_a x) * sigmoid(W_b x) ) ).
        attention_a: torch.Tensor = self._tanh(self._a(features))
        attention_b: torch.Tensor = self._sigmoid(self._b(features))
        gated: torch.Tensor = attention_a * attention_b
        gated = self._dropout(gated)

        logits: torch.Tensor = self._c(gated).squeeze(-1)
        # Exclude padded tokens before softmax.
        masked_logits: torch.Tensor = logits.masked_fill(~mask, float("-inf"))

        attention: torch.Tensor = torch.softmax(masked_logits, dim=1)
        attention = attention.masked_fill(~mask, 0.0)

        # Re-normalize for numerical stability.
        denom: torch.Tensor = attention.sum(dim=1, keepdim=True).clamp_min(self._eps)
        attention = attention / denom

        pooled: torch.Tensor = torch.einsum("bn,bnd->bd", attention, features)
        return pooled, attention


class ThreadsSlideEncoder(nn.Module):
    """THREADS slide encoder with gated attention and optional multi-head fusion.

    The encoder receives patch features and a patch-validity mask, then returns a
    1024-d slide embedding by default.
    """

    def __init__(
        self,
        in_dim: int = DEFAULT_INPUT_DIM,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        out_dim: int = DEFAULT_OUTPUT_DIM,
        n_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()

        validated_in_dim: int = self._validate_positive_int(in_dim, "in_dim")
        validated_hidden_dim: int = self._validate_positive_int(hidden_dim, "hidden_dim")
        validated_out_dim: int = self._validate_positive_int(out_dim, "out_dim")
        validated_heads: int = self._validate_positive_int(n_heads, "n_heads")
        validated_dropout: float = self._validate_dropout(dropout)

        self._in_dim: int = validated_in_dim
        self._hidden_dim: int = validated_hidden_dim
        self._out_dim: int = validated_out_dim
        self._n_heads: int = validated_heads
        self._dropout_p: float = validated_dropout

        # Paper/config expectations (strict defaults, but configurable for ablations).
        if self._out_dim != DEFAULT_OUTPUT_DIM:
            raise SlideEncoderConfigError(
                f"out_dim must be {DEFAULT_OUTPUT_DIM} for THREADS compatibility, got {self._out_dim}."
            )

        if self._n_heads == 1:
            self._pre_attention: nn.Module = self._build_single_head_pre_attention()
        else:
            self._pre_attention = self._build_multi_head_pre_attention()

        self._attention_heads: nn.ModuleList = nn.ModuleList(
            [_GatedAttentionHead(hidden_dim=self._hidden_dim, dropout=DEFAULT_ATTN_DROPOUT, eps=DEFAULT_EPS)
             for _ in range(self._n_heads)]
        )

        if self._n_heads > 1:
            self._post_attention_projection: nn.Linear = nn.Linear(
                self._n_heads * self._hidden_dim,
                self._out_dim,
            )
        else:
            # Keep explicit projection when hidden/out differ in future ablations.
            self._post_attention_projection = nn.Linear(self._hidden_dim, self._out_dim)

        self._last_attention_weights: Optional[torch.Tensor] = None
        self._last_shape: Optional[_ShapeInfo] = None

    def forward(self, patch_features: object, patch_mask: object) -> object:
        """Encode patch sequence into slide embedding.

        Args:
            patch_features: Tensor-like object with shape [B, N, in_dim].
            patch_mask: Tensor-like object with shape [B, N].

        Returns:
            Tensor with shape [B, out_dim].
        """
        features_tensor: torch.Tensor = self._coerce_patch_features(patch_features)
        mask_tensor: torch.Tensor = self._coerce_patch_mask(patch_mask, features_tensor)

        shape_info: _ShapeInfo = _ShapeInfo(
            batch_size=int(features_tensor.shape[0]),
            num_patches=int(features_tensor.shape[1]),
            feature_dim=int(features_tensor.shape[2]),
        )
        self._validate_non_empty_mask(mask_tensor, shape_info)

        projected: torch.Tensor = self._pre_attention(features_tensor)

        if self._n_heads == 1:
            head_features: torch.Tensor = projected.unsqueeze(1)  # [B,1,N,D]
        else:
            batch_size: int = int(projected.shape[0])
            num_patches: int = int(projected.shape[1])
            expected_width: int = self._n_heads * self._hidden_dim
            if int(projected.shape[2]) != expected_width:
                raise SlideEncoderRuntimeError(
                    "Projected feature width mismatch for multi-head mode: "
                    f"expected {expected_width}, got {int(projected.shape[2])}."
                )
            head_features = projected.view(batch_size, num_patches, self._n_heads, self._hidden_dim)
            head_features = head_features.permute(0, 2, 1, 3).contiguous()  # [B,H,N,D]

        pooled_per_head: list[torch.Tensor] = []
        attention_per_head: list[torch.Tensor] = []

        head_index: int
        for head_index in range(self._n_heads):
            pooled_head, attention_head = self._attention_heads[head_index](
                features=head_features[:, head_index, :, :],
                mask=mask_tensor,
            )
            pooled_per_head.append(pooled_head)
            attention_per_head.append(attention_head)

        pooled_stack: torch.Tensor = torch.stack(pooled_per_head, dim=1)  # [B,H,D]
        attention_stack: torch.Tensor = torch.stack(attention_per_head, dim=1)  # [B,H,N]

        if self._n_heads > 1:
            fused: torch.Tensor = pooled_stack.reshape(shape_info.batch_size, self._n_heads * self._hidden_dim)
        else:
            fused = pooled_stack[:, 0, :]

        slide_embedding: torch.Tensor = self._post_attention_projection(fused)

        if slide_embedding.ndim != 2 or int(slide_embedding.shape[1]) != self._out_dim:
            raise SlideEncoderRuntimeError(
                f"Output embedding shape mismatch: got {tuple(slide_embedding.shape)}, "
                f"expected [B,{self._out_dim}]."
            )

        # Store last attention in public contract shape.
        if self._n_heads == 1:
            self._last_attention_weights = attention_stack[:, 0, :].detach()
        else:
            self._last_attention_weights = attention_stack.detach()

        self._last_shape = shape_info
        return slide_embedding

    def attention_weights(self) -> object:
        """Return most recent forward-pass attention weights.

        Returns:
            - single-head: Tensor [B, N]
            - multi-head: Tensor [B, H, N]

        Raises:
            SlideEncoderRuntimeError: If called before first forward pass.
        """
        if self._last_attention_weights is None:
            raise SlideEncoderRuntimeError(
                "attention_weights() called before a successful forward() pass."
            )
        return self._last_attention_weights

    def _build_single_head_pre_attention(self) -> nn.Sequential:
        """Build single-head pre-attention projection network.

        Three-layer MLP with LayerNorm + GELU + dropout, ending in hidden_dim.
        """
        return nn.Sequential(
            nn.Linear(self._in_dim, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
        )

    def _build_multi_head_pre_attention(self) -> nn.Sequential:
        """Build multi-head pre-attention projection network.

        Final layer projects to n_heads * hidden_dim as described in the paper.
        """
        expanded_dim: int = self._n_heads * self._hidden_dim
        return nn.Sequential(
            nn.Linear(self._in_dim, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._hidden_dim, self._hidden_dim),
            nn.LayerNorm(self._hidden_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
            nn.Linear(self._hidden_dim, expanded_dim),
            nn.LayerNorm(expanded_dim),
            nn.GELU(),
            nn.Dropout(self._dropout_p),
        )

    def _coerce_patch_features(self, patch_features: object) -> torch.Tensor:
        """Validate and coerce patch features to float tensor [B,N,D]."""
        if not isinstance(patch_features, torch.Tensor):
            try:
                patch_features = torch.as_tensor(patch_features)
            except Exception as exc:  # noqa: BLE001
                raise SlideEncoderInputError(
                    f"patch_features cannot be converted to torch.Tensor: {exc}"
                ) from exc

        features_tensor: torch.Tensor = patch_features
        if features_tensor.ndim != 3:
            raise SlideEncoderInputError(
                f"patch_features must have shape [B,N,D], got {tuple(features_tensor.shape)}."
            )

        batch_size: int = int(features_tensor.shape[0])
        num_patches: int = int(features_tensor.shape[1])
        feature_dim: int = int(features_tensor.shape[2])

        if batch_size <= 0:
            raise SlideEncoderInputError("patch_features batch dimension must be > 0.")
        if num_patches <= 0:
            raise SlideEncoderInputError("patch_features patch dimension must be > 0.")
        if feature_dim != self._in_dim:
            raise SlideEncoderInputError(
                f"patch_features last dim mismatch: expected {self._in_dim}, got {feature_dim}."
            )

        if not torch.is_floating_point(features_tensor):
            features_tensor = features_tensor.to(torch.float32)
        else:
            features_tensor = features_tensor.to(torch.float32)

        if not torch.isfinite(features_tensor).all():
            raise SlideEncoderInputError("patch_features contains NaN or Inf values.")

        return features_tensor

    def _coerce_patch_mask(self, patch_mask: object, features_tensor: torch.Tensor) -> torch.Tensor:
        """Validate and coerce patch mask to bool tensor [B,N]."""
        if not isinstance(patch_mask, torch.Tensor):
            try:
                patch_mask = torch.as_tensor(patch_mask)
            except Exception as exc:  # noqa: BLE001
                raise SlideEncoderInputError(
                    f"patch_mask cannot be converted to torch.Tensor: {exc}"
                ) from exc

        mask_tensor: torch.Tensor = patch_mask
        if mask_tensor.ndim != 2:
            raise SlideEncoderInputError(
                f"patch_mask must have shape [B,N], got {tuple(mask_tensor.shape)}."
            )

        expected_shape: Tuple[int, int] = (int(features_tensor.shape[0]), int(features_tensor.shape[1]))
        actual_shape: Tuple[int, int] = (int(mask_tensor.shape[0]), int(mask_tensor.shape[1]))
        if actual_shape != expected_shape:
            raise SlideEncoderInputError(
                "patch_mask shape mismatch: "
                f"expected {expected_shape}, got {actual_shape}."
            )

        if mask_tensor.dtype == torch.bool:
            bool_mask: torch.Tensor = mask_tensor
        elif torch.is_floating_point(mask_tensor) or mask_tensor.dtype in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            bool_mask = mask_tensor > 0
        else:
            raise SlideEncoderInputError(
                f"Unsupported patch_mask dtype: {mask_tensor.dtype}."
            )

        return bool_mask

    def _validate_non_empty_mask(self, mask_tensor: torch.Tensor, shape_info: _ShapeInfo) -> None:
        """Ensure each sample has at least one valid patch."""
        valid_counts: torch.Tensor = mask_tensor.sum(dim=1)
        invalid_rows: torch.Tensor = torch.nonzero(valid_counts == 0, as_tuple=False).flatten()
        if int(invalid_rows.numel()) > 0:
            invalid_indices: list[int] = [int(value) for value in invalid_rows.tolist()]
            raise SlideEncoderInputError(
                "Each sample must have at least one valid patch in patch_mask. "
                f"Invalid batch rows={invalid_indices}; batch_shape="
                f"(B={shape_info.batch_size}, N={shape_info.num_patches}, D={shape_info.feature_dim})."
            )

    @staticmethod
    def _validate_positive_int(value: int, field_name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise SlideEncoderConfigError(
                f"{field_name} must be an int, got {type(value).__name__}."
            )
        if value <= 0:
            raise SlideEncoderConfigError(f"{field_name} must be > 0, got {value}.")
        return int(value)

    @staticmethod
    def _validate_dropout(value: float) -> float:
        if isinstance(value, bool):
            raise SlideEncoderConfigError("dropout must be float in [0,1), got bool.")
        try:
            dropout_value: float = float(value)
        except Exception as exc:  # noqa: BLE001
            raise SlideEncoderConfigError(f"dropout must be float, got {value!r}.") from exc

        if not (0.0 <= dropout_value < 1.0):
            raise SlideEncoderConfigError(
                f"dropout must be in [0,1), got {dropout_value}."
            )
        return dropout_value


__all__ = [
    "DEFAULT_INPUT_DIM",
    "DEFAULT_HIDDEN_DIM",
    "DEFAULT_OUTPUT_DIM",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_DROPOUT",
    "DEFAULT_ATTN_DROPOUT",
    "SlideEncoderError",
    "SlideEncoderConfigError",
    "SlideEncoderInputError",
    "SlideEncoderRuntimeError",
    "ThreadsSlideEncoder",
]
