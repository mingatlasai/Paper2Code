"""scGPT-style RNA encoder wrapper for THREADS.

This module implements the design-locked interface:
- ``ScGPTRNAEncoder.__init__(ckpt_path: str, out_dim: int, trainable: bool) -> None``
- ``ScGPTRNAEncoder.forward(gene_ids: object, expr_vals: object, gene_mask: object) -> object``

Paper/config alignment:
- Backbone depth: 12 transformer layers
- Attention heads: 8
- Output projection width: 1024
- Trainable/frozen mode controlled by ``trainable``

Notes:
- This implementation is intentionally self-contained and does not require an
  external ``scgpt`` package at runtime.
- If ``ckpt_path`` is provided, a best-effort state-dict load is performed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn


# -----------------------------------------------------------------------------
# Config-anchored defaults
# -----------------------------------------------------------------------------
DEFAULT_GENE_EMBED_DIM: int = 512
DEFAULT_EXPR_EMBED_DIM: int = 512
DEFAULT_MODEL_WIDTH: int = 512
DEFAULT_TRANSFORMER_LAYERS: int = 12
DEFAULT_ATTENTION_HEADS: int = 8
DEFAULT_OUTPUT_DIM: int = 1024

DEFAULT_EXPR_DROPOUT: float = 0.2
DEFAULT_EMBED_DROPOUT: float = 0.1
DEFAULT_TRANSFORMER_DROPOUT: float = 0.1
DEFAULT_MAX_GENES_FALLBACK: int = 65536
DEFAULT_LAYER_NORM_EPS: float = 1.0e-5


class RNAEncoderError(Exception):
    """Base exception for RNA encoder failures."""


class RNAEncoderConfigError(RNAEncoderError):
    """Raised when constructor configuration is invalid."""


class RNAEncoderInputError(RNAEncoderError):
    """Raised when forward inputs are invalid."""


class RNAEncoderCheckpointError(RNAEncoderError):
    """Raised when checkpoint loading fails."""


@dataclass(frozen=True)
class _ShapeInfo:
    """Validated shape metadata for one forward call."""

    batch_size: int
    seq_len: int


class ScGPTRNAEncoder(nn.Module):
    """scGPT-style RNA encoder with projection head to shared 1024-d space.

    Architecture follows the paper-level structure:
    - Gene identity encoder G (embedding + layer norm)
    - Expression value encoder E (2-layer MLP + layer norm, with input dropout)
    - Transformer encoder T (12 layers, 8 heads)
    - Mask-aware mean pooling (including CLS token)
    - Projection head P (2-layer MLP to out_dim)
    """

    def __init__(self, ckpt_path: str, out_dim: int, trainable: bool) -> None:
        """Initialize RNA encoder.

        Args:
            ckpt_path: Optional path to pretrained checkpoint.
            out_dim: Output embedding dimension. Must match 1024 for THREADS.
            trainable: If False, all parameters are frozen.
        """
        super().__init__()

        self._ckpt_path: str = "" if ckpt_path is None else str(ckpt_path).strip()
        self._out_dim: int = self._validate_out_dim(out_dim)
        self._trainable: bool = self._validate_trainable(trainable)

        self._gene_embed_dim: int = DEFAULT_GENE_EMBED_DIM
        self._expr_embed_dim: int = DEFAULT_EXPR_EMBED_DIM
        self._model_width: int = DEFAULT_MODEL_WIDTH
        self._layers: int = DEFAULT_TRANSFORMER_LAYERS
        self._heads: int = DEFAULT_ATTENTION_HEADS

        # G: gene identity encoder.
        self._gene_identity_encoder: nn.Embedding = nn.Embedding(
            num_embeddings=DEFAULT_MAX_GENES_FALLBACK,
            embedding_dim=self._gene_embed_dim,
        )
        self._gene_identity_norm: nn.LayerNorm = nn.LayerNorm(
            self._gene_embed_dim,
            eps=DEFAULT_LAYER_NORM_EPS,
        )

        # E: expression value encoder (1-d value -> 512-d token vector).
        self._expr_input_dropout: nn.Dropout = nn.Dropout(DEFAULT_EXPR_DROPOUT)
        self._expr_value_encoder: nn.Sequential = nn.Sequential(
            nn.Linear(1, self._expr_embed_dim),
            nn.GELU(),
            nn.Linear(self._expr_embed_dim, self._expr_embed_dim),
        )
        self._expr_value_norm: nn.LayerNorm = nn.LayerNorm(
            self._expr_embed_dim,
            eps=DEFAULT_LAYER_NORM_EPS,
        )

        # CLS token is part of pooled sequence.
        self._cls_token: nn.Parameter = nn.Parameter(
            torch.zeros(1, 1, self._model_width, dtype=torch.float32)
        )

        # T: transformer encoder.
        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self._model_width,
            nhead=self._heads,
            dim_feedforward=4 * self._model_width,
            dropout=DEFAULT_TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self._transformer: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self._layers,
            enable_nested_tensor=False,
        )
        self._token_dropout: nn.Dropout = nn.Dropout(DEFAULT_EMBED_DROPOUT)
        self._token_norm: nn.LayerNorm = nn.LayerNorm(
            self._model_width,
            eps=DEFAULT_LAYER_NORM_EPS,
        )

        # P: projection head to 1024.
        self._projection_head: nn.Sequential = nn.Sequential(
            nn.Linear(self._model_width, self._model_width),
            nn.GELU(),
            nn.Linear(self._model_width, self._out_dim),
        )

        self._reset_parameters()

        if self._ckpt_path:
            self._load_checkpoint(self._ckpt_path)

        self._apply_trainable_policy(self._trainable)

    def forward(self, gene_ids: object, expr_vals: object, gene_mask: object) -> object:
        """Encode RNA tokens to shared embedding space.

        Args:
            gene_ids: Tensor-like gene identifiers with shape [B, L].
            expr_vals: Tensor-like expression values with shape [B, L].
            gene_mask: Tensor-like validity mask with shape [B, L].

        Returns:
            Tensor of shape [B, out_dim].
        """
        gene_ids_tensor: torch.Tensor = self._coerce_gene_ids(gene_ids)
        expr_vals_tensor: torch.Tensor = self._coerce_expr_values(expr_vals)
        gene_mask_tensor: torch.Tensor = self._coerce_gene_mask(gene_mask)

        shape_info: _ShapeInfo = self._validate_shapes(
            gene_ids_tensor=gene_ids_tensor,
            expr_vals_tensor=expr_vals_tensor,
            gene_mask_tensor=gene_mask_tensor,
        )

        self._ensure_gene_vocab_capacity(gene_ids_tensor)

        # G: gene identity embeddings.
        gene_embed: torch.Tensor = self._gene_identity_encoder(gene_ids_tensor)
        gene_embed = self._gene_identity_norm(gene_embed)

        # E: expression-value embeddings.
        expr_input: torch.Tensor = expr_vals_tensor.unsqueeze(-1)
        expr_input = self._expr_input_dropout(expr_input)
        expr_embed: torch.Tensor = self._expr_value_encoder(expr_input)
        expr_embed = self._expr_value_norm(expr_embed)

        # Fuse G + E.
        token_embeddings: torch.Tensor = gene_embed + expr_embed

        # Prepend CLS and extend mask so pooling includes CLS.
        cls_token: torch.Tensor = self._cls_token.expand(shape_info.batch_size, -1, -1)
        token_embeddings = torch.cat([cls_token, token_embeddings], dim=1)
        token_embeddings = self._token_dropout(token_embeddings)

        cls_mask: torch.Tensor = torch.ones(
            (shape_info.batch_size, 1),
            dtype=torch.bool,
            device=gene_mask_tensor.device,
        )
        full_mask: torch.Tensor = torch.cat([cls_mask, gene_mask_tensor], dim=1)

        # PyTorch transformer expects True for padding positions.
        key_padding_mask: torch.Tensor = ~full_mask

        transformer_output: torch.Tensor = self._transformer(
            src=token_embeddings,
            src_key_padding_mask=key_padding_mask,
        )
        transformer_output = self._token_norm(transformer_output)

        pooled: torch.Tensor = self._masked_mean_pool(
            token_tensor=transformer_output,
            valid_mask=full_mask,
        )

        projected: torch.Tensor = self._projection_head(pooled)

        if projected.ndim != 2 or int(projected.shape[1]) != self._out_dim:
            raise RNAEncoderInputError(
                "Projection output shape mismatch: "
                f"expected [B,{self._out_dim}], got {tuple(projected.shape)}."
            )

        if not torch.isfinite(projected).all():
            raise RNAEncoderInputError("Non-finite values encountered in RNA embeddings.")

        return projected

    def _validate_out_dim(self, out_dim: int) -> int:
        if isinstance(out_dim, bool) or not isinstance(out_dim, int):
            raise RNAEncoderConfigError(
                f"out_dim must be int, got {type(out_dim).__name__}."
            )
        if out_dim <= 0:
            raise RNAEncoderConfigError(f"out_dim must be > 0, got {out_dim}.")
        if out_dim != DEFAULT_OUTPUT_DIM:
            raise RNAEncoderConfigError(
                f"THREADS RNA out_dim must be {DEFAULT_OUTPUT_DIM}, got {out_dim}."
            )
        return int(out_dim)

    def _validate_trainable(self, trainable: bool) -> bool:
        if not isinstance(trainable, bool):
            raise RNAEncoderConfigError(
                f"trainable must be bool, got {type(trainable).__name__}."
            )
        return bool(trainable)

    def _reset_parameters(self) -> None:
        # Conservative initialization for stable early training.
        nn.init.normal_(self._cls_token, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _load_checkpoint(self, ckpt_path: str) -> None:
        checkpoint_file: Path = Path(ckpt_path).expanduser().resolve()
        if not checkpoint_file.exists() or not checkpoint_file.is_file():
            raise RNAEncoderCheckpointError(f"Checkpoint file not found: {checkpoint_file}")

        try:
            checkpoint_obj: Any = torch.load(checkpoint_file, map_location="cpu")
        except Exception as exc:  # noqa: BLE001
            raise RNAEncoderCheckpointError(
                f"Failed to load checkpoint at {checkpoint_file}: {exc}"
            ) from exc

        state_dict: Dict[str, torch.Tensor] = self._extract_state_dict(checkpoint_obj)
        cleaned_state_dict: Dict[str, torch.Tensor] = self._clean_state_dict_keys(state_dict)

        try:
            self.load_state_dict(cleaned_state_dict, strict=False)
        except Exception as exc:  # noqa: BLE001
            raise RNAEncoderCheckpointError(
                "Failed to load checkpoint state_dict into ScGPTRNAEncoder: "
                f"{exc}"
            ) from exc

    def _extract_state_dict(self, checkpoint_obj: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_obj, Mapping):
            if "state_dict" in checkpoint_obj and isinstance(checkpoint_obj["state_dict"], Mapping):
                return {str(key): value for key, value in checkpoint_obj["state_dict"].items()}
            if "model" in checkpoint_obj and isinstance(checkpoint_obj["model"], Mapping):
                return {str(key): value for key, value in checkpoint_obj["model"].items()}
            # Direct state dict case.
            if all(isinstance(key, str) for key in checkpoint_obj.keys()):
                return {str(key): value for key, value in checkpoint_obj.items()}

        raise RNAEncoderCheckpointError(
            "Unsupported checkpoint format. Expected mapping with 'state_dict' or model weights."
        )

    def _clean_state_dict_keys(self, state_dict: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        cleaned: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            normalized_key: str = str(key)
            for prefix in (
                "module.",
                "model.",
                "rna_encoder.",
                "encoder.",
                "scgpt.",
            ):
                if normalized_key.startswith(prefix):
                    normalized_key = normalized_key[len(prefix) :]
            cleaned[normalized_key] = value
        return cleaned

    def _apply_trainable_policy(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)

    def _coerce_gene_ids(self, gene_ids: object) -> torch.Tensor:
        try:
            tensor_value: torch.Tensor = torch.as_tensor(gene_ids)
        except Exception as exc:  # noqa: BLE001
            raise RNAEncoderInputError(f"gene_ids cannot be converted to tensor: {exc}") from exc

        if tensor_value.ndim != 2:
            raise RNAEncoderInputError(
                f"gene_ids must have shape [B,L], got {tuple(tensor_value.shape)}."
            )

        if tensor_value.dtype == torch.bool:
            raise RNAEncoderInputError("gene_ids cannot be bool tensor.")

        tensor_value = tensor_value.to(dtype=torch.long)
        if (tensor_value < 0).any():
            raise RNAEncoderInputError("gene_ids contains negative indices.")

        return tensor_value

    def _coerce_expr_values(self, expr_vals: object) -> torch.Tensor:
        try:
            tensor_value: torch.Tensor = torch.as_tensor(expr_vals)
        except Exception as exc:  # noqa: BLE001
            raise RNAEncoderInputError(f"expr_vals cannot be converted to tensor: {exc}") from exc

        if tensor_value.ndim != 2:
            raise RNAEncoderInputError(
                f"expr_vals must have shape [B,L], got {tuple(tensor_value.shape)}."
            )

        tensor_value = tensor_value.to(dtype=torch.float32)
        if not torch.isfinite(tensor_value).all():
            raise RNAEncoderInputError("expr_vals contains NaN/Inf.")
        return tensor_value

    def _coerce_gene_mask(self, gene_mask: object) -> torch.Tensor:
        try:
            tensor_value: torch.Tensor = torch.as_tensor(gene_mask)
        except Exception as exc:  # noqa: BLE001
            raise RNAEncoderInputError(f"gene_mask cannot be converted to tensor: {exc}") from exc

        if tensor_value.ndim != 2:
            raise RNAEncoderInputError(
                f"gene_mask must have shape [B,L], got {tuple(tensor_value.shape)}."
            )

        if tensor_value.dtype == torch.bool:
            return tensor_value

        if torch.is_floating_point(tensor_value):
            return tensor_value > 0.5

        return tensor_value > 0

    def _validate_shapes(
        self,
        gene_ids_tensor: torch.Tensor,
        expr_vals_tensor: torch.Tensor,
        gene_mask_tensor: torch.Tensor,
    ) -> _ShapeInfo:
        if gene_ids_tensor.shape != expr_vals_tensor.shape:
            raise RNAEncoderInputError(
                "gene_ids and expr_vals shape mismatch: "
                f"{tuple(gene_ids_tensor.shape)} vs {tuple(expr_vals_tensor.shape)}."
            )
        if gene_ids_tensor.shape != gene_mask_tensor.shape:
            raise RNAEncoderInputError(
                "gene_ids and gene_mask shape mismatch: "
                f"{tuple(gene_ids_tensor.shape)} vs {tuple(gene_mask_tensor.shape)}."
            )

        batch_size: int = int(gene_ids_tensor.shape[0])
        seq_len: int = int(gene_ids_tensor.shape[1])

        if batch_size <= 0:
            raise RNAEncoderInputError("Batch size must be > 0.")
        if seq_len <= 0:
            raise RNAEncoderInputError("Sequence length must be > 0.")

        valid_counts: torch.Tensor = gene_mask_tensor.sum(dim=1)
        if (valid_counts <= 0).any():
            raise RNAEncoderInputError(
                "Each sample must contain at least one valid gene token in gene_mask."
            )

        return _ShapeInfo(batch_size=batch_size, seq_len=seq_len)

    def _ensure_gene_vocab_capacity(self, gene_ids_tensor: torch.Tensor) -> None:
        max_gene_index: int = int(torch.max(gene_ids_tensor).item())
        current_vocab: int = int(self._gene_identity_encoder.num_embeddings)
        if max_gene_index < current_vocab:
            return

        new_vocab: int = max(current_vocab, max_gene_index + 1)
        new_embedding: nn.Embedding = nn.Embedding(
            num_embeddings=new_vocab,
            embedding_dim=self._gene_embed_dim,
            device=self._gene_identity_encoder.weight.device,
            dtype=self._gene_identity_encoder.weight.dtype,
        )

        nn.init.normal_(new_embedding.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            new_embedding.weight[:current_vocab] = self._gene_identity_encoder.weight

        new_embedding.weight.requires_grad = self._gene_identity_encoder.weight.requires_grad
        self._gene_identity_encoder = new_embedding

    def _masked_mean_pool(self, token_tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if token_tensor.ndim != 3:
            raise RNAEncoderInputError(
                f"token_tensor must be [B,L,D], got {tuple(token_tensor.shape)}."
            )
        if valid_mask.ndim != 2:
            raise RNAEncoderInputError(
                f"valid_mask must be [B,L], got {tuple(valid_mask.shape)}."
            )
        if token_tensor.shape[:2] != valid_mask.shape:
            raise RNAEncoderInputError(
                "token_tensor and valid_mask shape mismatch: "
                f"{tuple(token_tensor.shape[:2])} vs {tuple(valid_mask.shape)}."
            )

        mask_float: torch.Tensor = valid_mask.to(dtype=token_tensor.dtype).unsqueeze(-1)
        summed: torch.Tensor = (token_tensor * mask_float).sum(dim=1)
        denom: torch.Tensor = mask_float.sum(dim=1).clamp_min(1.0)
        pooled: torch.Tensor = summed / denom

        if not torch.isfinite(pooled).all():
            raise RNAEncoderInputError("Non-finite pooled RNA token representation.")

        return pooled


__all__ = [
    "DEFAULT_GENE_EMBED_DIM",
    "DEFAULT_EXPR_EMBED_DIM",
    "DEFAULT_MODEL_WIDTH",
    "DEFAULT_TRANSFORMER_LAYERS",
    "DEFAULT_ATTENTION_HEADS",
    "DEFAULT_OUTPUT_DIM",
    "RNAEncoderError",
    "RNAEncoderConfigError",
    "RNAEncoderInputError",
    "RNAEncoderCheckpointError",
    "ScGPTRNAEncoder",
]
