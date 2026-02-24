"""2D ALiBi attention bias for TITAN.

This module implements Euclidean-distance ALiBi for 2D patch-grid coordinates.
It follows the design-locked public interface:
- ``ALiBi2D.__init__(num_heads: int, slopes: torch.Tensor)``
- ``ALiBi2D.build_bias(coords_xy: torch.Tensor) -> torch.Tensor``
- ``ALiBi2D.apply(attn_scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor``
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Config-locked constants from config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE_PX: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768

_NUM_HEADS: int = 12
_HEAD_DIM: int = 64
_EMBED_DIM: int = 768
_MLP_DIM: int = 3072


class ALiBi2DError(RuntimeError):
    """Base exception for ALiBi2D-specific runtime failures."""


class ALiBi2D(nn.Module):
    """2D Euclidean ALiBi bias builder and applier.

    Let ``m_h`` be the slope for attention head ``h``, and let
    ``d(i, j)`` be Euclidean distance between tokens ``i`` and ``j`` in 2D
    patch-grid coordinates. The per-head bias is:

    ``bias_h(i, j) = -m_h * d(i, j)``

    The returned bias is additive and intended to be added to pre-softmax
    attention scores.
    """

    def __init__(self, num_heads: int, slopes: torch.Tensor) -> None:
        """Initialize ALiBi2D.

        Args:
            num_heads: Number of attention heads.
            slopes: Tensor of shape ``[num_heads]`` containing non-negative
                finite slope values.
        """
        super().__init__()

        if isinstance(num_heads, bool) or not isinstance(num_heads, int):
            raise TypeError(f"num_heads must be int, got {type(num_heads).__name__}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")

        if not isinstance(slopes, torch.Tensor):
            raise TypeError(f"slopes must be torch.Tensor, got {type(slopes).__name__}.")

        slopes_flat: torch.Tensor = slopes.detach().flatten()
        if slopes_flat.ndim != 1:
            raise ValueError(f"slopes must be rank-1 after flatten, got shape {tuple(slopes.shape)}.")
        if int(slopes_flat.shape[0]) != num_heads:
            raise ValueError(
                f"slopes length must equal num_heads ({num_heads}), got {int(slopes_flat.shape[0])}."
            )
        if not torch.isfinite(slopes_flat).all():
            raise ValueError("slopes contains non-finite values.")
        if (slopes_flat < 0).any():
            raise ValueError("slopes must be non-negative.")

        self.num_heads: int = int(num_heads)
        self.patch_size_px: int = _PATCH_SIZE_PX
        self.magnification: str = _MAGNIFICATION

        # Keep slopes in float32 for numerically stable distance scaling.
        self.register_buffer("slopes", slopes_flat.to(dtype=torch.float32), persistent=True)

    def build_bias(self, coords_xy: torch.Tensor) -> torch.Tensor:
        """Build 2D ALiBi bias from token coordinates.

        Args:
            coords_xy: Coordinates in one of the following shapes:
                - ``[T, 2]``
                - ``[B, T, 2]``

        Returns:
            Bias tensor:
                - ``[H, T, T]`` for unbatched input
                - ``[B, H, T, T]`` for batched input
        """
        if not isinstance(coords_xy, torch.Tensor):
            raise TypeError(f"coords_xy must be torch.Tensor, got {type(coords_xy).__name__}.")
        if coords_xy.ndim not in (2, 3):
            raise ValueError(
                "coords_xy must have shape [T,2] or [B,T,2], "
                f"got {tuple(coords_xy.shape)}."
            )
        if int(coords_xy.shape[-1]) != 2:
            raise ValueError(
                f"coords_xy last dimension must be 2, got {int(coords_xy.shape[-1])}."
            )

        is_unbatched: bool = coords_xy.ndim == 2
        coords: torch.Tensor = coords_xy.unsqueeze(0) if is_unbatched else coords_xy

        batch_size: int = int(coords.shape[0])
        num_tokens: int = int(coords.shape[1])
        if batch_size <= 0 or num_tokens <= 0:
            raise ValueError(
                f"coords_xy has invalid shape {tuple(coords_xy.shape)} with non-positive batch/tokens."
            )

        coords_f32: torch.Tensor = coords.to(dtype=torch.float32)

        # Pairwise Euclidean distances in 2D.
        # diff shape: [B, T, T, 2]
        diff: torch.Tensor = coords_f32[:, :, None, :] - coords_f32[:, None, :, :]
        dist: torch.Tensor = torch.linalg.norm(diff, ord=2, dim=-1)  # [B, T, T]

        slopes: torch.Tensor = self.slopes.to(device=coords_f32.device, dtype=coords_f32.dtype)
        bias_bhtt: torch.Tensor = -slopes.view(1, self.num_heads, 1, 1) * dist.unsqueeze(1)

        if is_unbatched:
            return bias_bhtt.squeeze(0)
        return bias_bhtt

    def apply(self, attn_scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Apply additive ALiBi bias to attention scores.

        Args:
            attn_scores: Attention score tensor with shape ``[B, H, T, T]``.
            bias: Bias tensor with shape ``[H, T, T]`` or ``[B, H, T, T]``.

        Returns:
            Tensor with same shape as ``attn_scores``.
        """
        if not isinstance(attn_scores, torch.Tensor):
            raise TypeError(f"attn_scores must be torch.Tensor, got {type(attn_scores).__name__}.")
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"bias must be torch.Tensor, got {type(bias).__name__}.")

        if attn_scores.ndim != 4:
            raise ValueError(
                f"attn_scores must have shape [B,H,T,T], got {tuple(attn_scores.shape)}."
            )

        batch_size: int = int(attn_scores.shape[0])
        num_heads: int = int(attn_scores.shape[1])
        q_tokens: int = int(attn_scores.shape[2])
        k_tokens: int = int(attn_scores.shape[3])

        if q_tokens != k_tokens:
            raise ValueError(
                "attn_scores last two dims must be square [T,T], "
                f"got ({q_tokens}, {k_tokens})."
            )
        if num_heads != self.num_heads:
            raise ValueError(
                f"attn_scores head dim must equal initialized num_heads={self.num_heads}, got {num_heads}."
            )

        bias_ready: torch.Tensor
        if bias.ndim == 3:
            h_dim: int = int(bias.shape[0])
            t_q: int = int(bias.shape[1])
            t_k: int = int(bias.shape[2])
            if h_dim != num_heads or t_q != q_tokens or t_k != k_tokens:
                raise ValueError(
                    "3D bias must have shape [H,T,T] matching attention, "
                    f"got {tuple(bias.shape)} vs expected ({num_heads},{q_tokens},{k_tokens})."
                )
            bias_ready = bias.unsqueeze(0)
        elif bias.ndim == 4:
            b_dim: int = int(bias.shape[0])
            h_dim = int(bias.shape[1])
            t_q = int(bias.shape[2])
            t_k = int(bias.shape[3])
            if h_dim != num_heads or t_q != q_tokens or t_k != k_tokens:
                raise ValueError(
                    "4D bias must match [B,H,T,T] (or B=1 for broadcast), "
                    f"got {tuple(bias.shape)} vs attention {tuple(attn_scores.shape)}."
                )
            if b_dim not in (1, batch_size):
                raise ValueError(
                    f"4D bias batch dim must be 1 or {batch_size}, got {b_dim}."
                )
            bias_ready = bias
        else:
            raise ValueError(
                f"bias must have shape [H,T,T] or [B,H,T,T], got {tuple(bias.shape)}."
            )

        bias_cast: torch.Tensor = bias_ready.to(device=attn_scores.device, dtype=attn_scores.dtype)
        return attn_scores + bias_cast


__all__ = [
    "ALiBi2DError",
    "ALiBi2D",
]
