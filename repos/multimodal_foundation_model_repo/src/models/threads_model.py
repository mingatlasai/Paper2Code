"""THREADS multimodal model composition.

This module implements the design-locked interface:
- ``ThreadsModel.__init__(patch_encoder, slide_encoder, rna_encoder, dna_encoder, loss_fn)``
- ``ThreadsModel.forward_wsi(patches, patch_mask)``
- ``ThreadsModel.forward_rna(gene_ids, expr_vals, gene_mask)``
- ``ThreadsModel.forward_dna(dna_multi_hot)``
- ``ThreadsModel.training_step(batch)``
- ``ThreadsModel.extract_slide_embedding(batch)``

Paper/config alignment:
- Shared embedding dimension: 1024
- DNA input dimension: 1673
- Contrastive objective: InfoNCE-style via injected ``ContrastiveLoss``
- Training uses available modality pairs (WSI↔RNA and/or WSI↔DNA)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from src.models.dna_encoder_mlp import DNAMLPEncoder, DEFAULT_INPUT_DIM as DNA_DEFAULT_INPUT_DIM
from src.models.losses import ContrastiveLoss
from src.models.patch_encoder import PatchEncoder
from src.models.rna_encoder_scgpt import ScGPTRNAEncoder
from src.models.slide_encoder_threads import ThreadsSlideEncoder


# -----------------------------------------------------------------------------
# Config-anchored constants
# -----------------------------------------------------------------------------
DEFAULT_EMBEDDING_DIM: int = 1024
DEFAULT_DNA_INPUT_DIM: int = DNA_DEFAULT_INPUT_DIM
DEFAULT_MODALITY_WEIGHT_RNA: float = 1.0
DEFAULT_MODALITY_WEIGHT_DNA: float = 1.0
DEFAULT_VALIDATE_NUMERICS: bool = True
DEFAULT_SKIP_MISSING_MODALITY: bool = True
DEFAULT_EPS: float = 1.0e-12

# Batch keys (aligned with src/data/datamodules.py collate outputs).
BATCH_KEY_PATCH_FEATURES: str = "patch_features"
BATCH_KEY_PATCH_MASK: str = "patch_mask"
BATCH_KEY_SAMPLE_ID: str = "sample_id"

BATCH_KEY_HAS_RNA: str = "has_rna"
BATCH_KEY_GENE_IDS: str = "gene_ids"
BATCH_KEY_EXPR_VALS: str = "expr_vals"
BATCH_KEY_GENE_MASK: str = "gene_mask"

BATCH_KEY_HAS_DNA: str = "has_dna"
BATCH_KEY_DNA_MULTI_HOT: str = "dna_multi_hot"


class ThreadsModelError(Exception):
    """Base exception for THREADS composition failures."""


class ThreadsModelConfigError(ThreadsModelError):
    """Raised when constructor contracts are violated."""


class ThreadsModelInputError(ThreadsModelError):
    """Raised when forward/training inputs are malformed."""


class ThreadsModelRuntimeError(ThreadsModelError):
    """Raised when runtime loss/embedding behavior is invalid."""


@dataclass(frozen=True)
class _ForwardShape:
    """Validated shape metadata for one WSI forward pass."""

    batch_size: int
    num_patches: int
    feature_dim: int


class ThreadsModel(nn.Module):
    """Central THREADS multimodal model.

    This class composes patch->slide and molecular branches in a single object for:
    - pretraining contrastive loss computation
    - slide embedding extraction
    """

    def __init__(
        self,
        patch_encoder: PatchEncoder,
        slide_encoder: ThreadsSlideEncoder,
        rna_encoder: ScGPTRNAEncoder,
        dna_encoder: DNAMLPEncoder,
        loss_fn: ContrastiveLoss,
    ) -> None:
        """Initialize model composition with strict interface validation."""
        super().__init__()

        self._patch_encoder: PatchEncoder = self._validate_patch_encoder(patch_encoder)
        self._slide_encoder: ThreadsSlideEncoder = self._validate_slide_encoder(slide_encoder)
        self._rna_encoder: ScGPTRNAEncoder = self._validate_rna_encoder(rna_encoder)
        self._dna_encoder: DNAMLPEncoder = self._validate_dna_encoder(dna_encoder)
        self._loss_fn: ContrastiveLoss = self._validate_loss_fn(loss_fn)

        self._embedding_dim: int = DEFAULT_EMBEDDING_DIM
        self._dna_input_dim: int = DEFAULT_DNA_INPUT_DIM

        self._weight_rna: float = DEFAULT_MODALITY_WEIGHT_RNA
        self._weight_dna: float = DEFAULT_MODALITY_WEIGHT_DNA
        self._skip_missing_modality: bool = DEFAULT_SKIP_MISSING_MODALITY
        self._validate_numerics: bool = DEFAULT_VALIDATE_NUMERICS

        self._last_slide_embedding: Optional[torch.Tensor] = None
        self._last_training_outputs: Dict[str, float] = {}

        self._validate_branch_dim_compatibility()

    def forward_wsi(self, patches: object, patch_mask: object) -> object:
        """Encode WSI patches to slide embeddings.

        Args:
            patches: Tensor-like patch feature payload. Preferred shape [B, N, F].
                If rank-2 [N, F], batch dimension is added.
                If not feature-shaped, ``patch_encoder.encode`` is attempted.
            patch_mask: Tensor-like validity mask with shape [B, N] or [N].

        Returns:
            Tensor with shape [B, 1024].
        """
        patch_features: torch.Tensor = self._coerce_or_encode_patch_features(patches)
        mask_tensor: torch.Tensor = self._coerce_patch_mask(patch_mask, patch_features)

        shape_info: _ForwardShape = _ForwardShape(
            batch_size=int(patch_features.shape[0]),
            num_patches=int(patch_features.shape[1]),
            feature_dim=int(patch_features.shape[2]),
        )
        self._validate_mask_has_valid_tokens(mask_tensor, shape_info)

        slide_embedding: torch.Tensor = self._slide_encoder(
            patch_features=patch_features,
            patch_mask=mask_tensor,
        )
        if not isinstance(slide_embedding, torch.Tensor):
            raise ThreadsModelRuntimeError(
                "slide_encoder.forward returned non-tensor output."
            )
        if slide_embedding.ndim != 2:
            raise ThreadsModelRuntimeError(
                f"slide embedding must be rank-2 [B,D], got shape={tuple(slide_embedding.shape)}."
            )
        if int(slide_embedding.shape[1]) != self._embedding_dim:
            raise ThreadsModelRuntimeError(
                "slide embedding width mismatch: "
                f"expected {self._embedding_dim}, got {int(slide_embedding.shape[1])}."
            )

        if self._validate_numerics and not torch.isfinite(slide_embedding).all():
            raise ThreadsModelRuntimeError("slide embedding contains NaN/Inf.")

        self._last_slide_embedding = slide_embedding.detach()
        return slide_embedding

    def forward_rna(self, gene_ids: object, expr_vals: object, gene_mask: object) -> object:
        """Encode RNA payload to shared embedding space [B, 1024]."""
        rna_embedding: Any = self._rna_encoder(
            gene_ids=gene_ids,
            expr_vals=expr_vals,
            gene_mask=gene_mask,
        )
        if not isinstance(rna_embedding, torch.Tensor):
            raise ThreadsModelRuntimeError("rna_encoder.forward returned non-tensor output.")
        if rna_embedding.ndim != 2:
            raise ThreadsModelRuntimeError(
                f"RNA embedding must be rank-2 [B,D], got shape={tuple(rna_embedding.shape)}."
            )
        if int(rna_embedding.shape[1]) != self._embedding_dim:
            raise ThreadsModelRuntimeError(
                "RNA embedding width mismatch: "
                f"expected {self._embedding_dim}, got {int(rna_embedding.shape[1])}."
            )
        if self._validate_numerics and not torch.isfinite(rna_embedding).all():
            raise ThreadsModelRuntimeError("RNA embedding contains NaN/Inf.")
        return rna_embedding

    def forward_dna(self, dna_multi_hot: object) -> object:
        """Encode DNA multi-hot vectors to shared embedding space [B, 1024]."""
        dna_embedding: Any = self._dna_encoder(dna_multi_hot=dna_multi_hot)
        if not isinstance(dna_embedding, torch.Tensor):
            raise ThreadsModelRuntimeError("dna_encoder.forward returned non-tensor output.")
        if dna_embedding.ndim != 2:
            raise ThreadsModelRuntimeError(
                f"DNA embedding must be rank-2 [B,D], got shape={tuple(dna_embedding.shape)}."
            )
        if int(dna_embedding.shape[1]) != self._embedding_dim:
            raise ThreadsModelRuntimeError(
                "DNA embedding width mismatch: "
                f"expected {self._embedding_dim}, got {int(dna_embedding.shape[1])}."
            )
        if self._validate_numerics and not torch.isfinite(dna_embedding).all():
            raise ThreadsModelRuntimeError("DNA embedding contains NaN/Inf.")
        return dna_embedding

    def training_step(self, batch: dict[str, object]) -> dict[str, float]:
        """Compute multimodal contrastive loss for one pretraining batch.

        The method uses available modality pairs:
        - WSI↔RNA when RNA is available
        - WSI↔DNA when DNA is available

        Returns:
            Dictionary with scalar float metrics, including:
            - ``loss`` / ``loss_total``
            - ``loss_rna`` (if available)
            - ``loss_dna`` (if available)
            - ``n_pairs_rna``
            - ``n_pairs_dna``
        """
        if not isinstance(batch, Mapping):
            raise ThreadsModelInputError(
                f"training_step expects mapping batch, got {type(batch).__name__}."
            )

        patch_payload: object = self._require_key(batch, BATCH_KEY_PATCH_FEATURES)
        patch_mask_payload: object = self._require_key(batch, BATCH_KEY_PATCH_MASK)

        z_wsi: torch.Tensor = self.forward_wsi(
            patches=patch_payload,
            patch_mask=patch_mask_payload,
        )
        batch_size: int = int(z_wsi.shape[0])

        outputs: Dict[str, float] = {
            "n_pairs_rna": 0.0,
            "n_pairs_dna": 0.0,
        }

        loss_terms: List[torch.Tensor] = []
        loss_weights: List[float] = []

        # ------------------------------------------------------------------
        # RNA branch pairing
        # ------------------------------------------------------------------
        rna_pair_indices: torch.Tensor = self._resolve_modality_indices(
            batch=batch,
            modality_flag_key=BATCH_KEY_HAS_RNA,
            default_size=batch_size,
        )
        if int(rna_pair_indices.numel()) > 0:
            gene_ids: object = self._require_key(batch, BATCH_KEY_GENE_IDS)
            expr_vals: object = self._require_key(batch, BATCH_KEY_EXPR_VALS)
            gene_mask: object = self._require_key(batch, BATCH_KEY_GENE_MASK)

            gene_ids_tensor: torch.Tensor = self._coerce_tensor(gene_ids, name=BATCH_KEY_GENE_IDS)
            expr_vals_tensor: torch.Tensor = self._coerce_tensor(expr_vals, name=BATCH_KEY_EXPR_VALS)
            gene_mask_tensor: torch.Tensor = self._coerce_tensor(gene_mask, name=BATCH_KEY_GENE_MASK)

            self._validate_first_dim(gene_ids_tensor, batch_size, BATCH_KEY_GENE_IDS)
            self._validate_first_dim(expr_vals_tensor, batch_size, BATCH_KEY_EXPR_VALS)
            self._validate_first_dim(gene_mask_tensor, batch_size, BATCH_KEY_GENE_MASK)

            z_wsi_rna: torch.Tensor = z_wsi.index_select(dim=0, index=rna_pair_indices)
            z_rna: torch.Tensor = self.forward_rna(
                gene_ids=gene_ids_tensor.index_select(dim=0, index=rna_pair_indices),
                expr_vals=expr_vals_tensor.index_select(dim=0, index=rna_pair_indices),
                gene_mask=gene_mask_tensor.index_select(dim=0, index=rna_pair_indices),
            )
            loss_rna_tensor: torch.Tensor = self._loss_fn(z_wsi=z_wsi_rna, z_mol=z_rna)
            self._validate_loss_tensor(loss_rna_tensor, name="loss_rna")

            outputs["loss_rna"] = float(loss_rna_tensor.detach().cpu().item())
            outputs["n_pairs_rna"] = float(int(rna_pair_indices.numel()))

            loss_terms.append(loss_rna_tensor)
            loss_weights.append(float(self._weight_rna))

        # ------------------------------------------------------------------
        # DNA branch pairing
        # ------------------------------------------------------------------
        dna_pair_indices: torch.Tensor = self._resolve_modality_indices(
            batch=batch,
            modality_flag_key=BATCH_KEY_HAS_DNA,
            default_size=batch_size,
        )
        if int(dna_pair_indices.numel()) > 0:
            dna_multi_hot: object = self._require_key(batch, BATCH_KEY_DNA_MULTI_HOT)
            dna_tensor: torch.Tensor = self._coerce_tensor(
                dna_multi_hot,
                name=BATCH_KEY_DNA_MULTI_HOT,
            )
            self._validate_first_dim(dna_tensor, batch_size, BATCH_KEY_DNA_MULTI_HOT)
            self._validate_dna_shape(dna_tensor)

            z_wsi_dna: torch.Tensor = z_wsi.index_select(dim=0, index=dna_pair_indices)
            z_dna: torch.Tensor = self.forward_dna(
                dna_multi_hot=dna_tensor.index_select(dim=0, index=dna_pair_indices),
            )
            loss_dna_tensor: torch.Tensor = self._loss_fn(z_wsi=z_wsi_dna, z_mol=z_dna)
            self._validate_loss_tensor(loss_dna_tensor, name="loss_dna")

            outputs["loss_dna"] = float(loss_dna_tensor.detach().cpu().item())
            outputs["n_pairs_dna"] = float(int(dna_pair_indices.numel()))

            loss_terms.append(loss_dna_tensor)
            loss_weights.append(float(self._weight_dna))

        if len(loss_terms) == 0:
            if self._skip_missing_modality:
                raise ThreadsModelInputError(
                    "No valid molecular pairs in batch; cannot compute contrastive loss. "
                    "Expected at least one of WSI↔RNA or WSI↔DNA."
                )
            raise ThreadsModelInputError("No modality pairs available in batch.")

        total_loss_tensor: torch.Tensor = self._weighted_mean_loss(loss_terms, loss_weights)
        self._validate_loss_tensor(total_loss_tensor, name="loss_total")

        total_loss_value: float = float(total_loss_tensor.detach().cpu().item())
        outputs["loss_total"] = total_loss_value
        outputs["loss"] = total_loss_value

        self._last_training_outputs = dict(outputs)
        return outputs

    def extract_slide_embedding(self, batch: dict[str, object]) -> object:
        """Extract slide embeddings for evaluation/export.

        Args:
            batch: Mapping that contains ``patch_features`` (or ``patches``) and
                ``patch_mask``.

        Returns:
            Tensor [B, 1024] slide embeddings.
        """
        if not isinstance(batch, Mapping):
            raise ThreadsModelInputError(
                f"extract_slide_embedding expects mapping batch, got {type(batch).__name__}."
            )

        patch_payload: object
        if BATCH_KEY_PATCH_FEATURES in batch:
            patch_payload = batch[BATCH_KEY_PATCH_FEATURES]
        elif "patches" in batch:
            patch_payload = batch["patches"]
        else:
            raise ThreadsModelInputError(
                "extract_slide_embedding requires 'patch_features' or 'patches' in batch."
            )

        patch_mask_payload: object = self._require_key(batch, BATCH_KEY_PATCH_MASK)

        with torch.inference_mode():
            embedding: torch.Tensor = self.forward_wsi(
                patches=patch_payload,
                patch_mask=patch_mask_payload,
            )

        return embedding

    def _validate_branch_dim_compatibility(self) -> None:
        """Validate static branch dimensions against config invariants."""
        patch_feature_dim: int = self._patch_encoder.feature_dim()
        if patch_feature_dim <= 0:
            raise ThreadsModelConfigError(
                f"patch_encoder.feature_dim() must be > 0, got {patch_feature_dim}."
            )

        slide_out_dim: Optional[int] = self._get_optional_int_attr(self._slide_encoder, "_out_dim")
        if slide_out_dim is not None and int(slide_out_dim) != self._embedding_dim:
            raise ThreadsModelConfigError(
                "slide encoder output dim mismatch: "
                f"expected {self._embedding_dim}, got {slide_out_dim}."
            )

        rna_out_dim: Optional[int] = self._get_optional_int_attr(self._rna_encoder, "_out_dim")
        if rna_out_dim is not None and int(rna_out_dim) != self._embedding_dim:
            raise ThreadsModelConfigError(
                f"RNA encoder output dim mismatch: expected {self._embedding_dim}, got {rna_out_dim}."
            )

        dna_out_dim: Optional[int]
        if hasattr(self._dna_encoder, "output_dim") and callable(self._dna_encoder.output_dim):
            dna_out_dim = int(self._dna_encoder.output_dim())
        else:
            dna_out_dim = self._get_optional_int_attr(self._dna_encoder, "_out_dim")

        if dna_out_dim is not None and int(dna_out_dim) != self._embedding_dim:
            raise ThreadsModelConfigError(
                f"DNA encoder output dim mismatch: expected {self._embedding_dim}, got {dna_out_dim}."
            )

        dna_in_dim: Optional[int]
        if hasattr(self._dna_encoder, "input_dim") and callable(self._dna_encoder.input_dim):
            dna_in_dim = int(self._dna_encoder.input_dim())
        else:
            dna_in_dim = self._get_optional_int_attr(self._dna_encoder, "_in_dim")

        if dna_in_dim is not None and int(dna_in_dim) != self._dna_input_dim:
            raise ThreadsModelConfigError(
                f"DNA encoder input dim mismatch: expected {self._dna_input_dim}, got {dna_in_dim}."
            )

    def _coerce_or_encode_patch_features(self, patches: object) -> torch.Tensor:
        """Coerce input to feature tensor [B,N,F], encoding raw patches if needed."""
        # Fast path for tensor-like features.
        if isinstance(patches, torch.Tensor):
            tensor_value: torch.Tensor = patches
        else:
            try:
                tensor_value = torch.as_tensor(patches)
            except Exception:
                tensor_value = None  # type: ignore[assignment]

        feature_dim: int = int(self._patch_encoder.feature_dim())

        if isinstance(tensor_value, torch.Tensor):
            if tensor_value.ndim == 3 and int(tensor_value.shape[2]) == feature_dim:
                output: torch.Tensor = tensor_value.to(torch.float32)
                if self._validate_numerics and not torch.isfinite(output).all():
                    raise ThreadsModelInputError("patch feature tensor contains NaN/Inf.")
                return output

            if tensor_value.ndim == 2 and int(tensor_value.shape[1]) == feature_dim:
                output = tensor_value.unsqueeze(0).to(torch.float32)
                if self._validate_numerics and not torch.isfinite(output).all():
                    raise ThreadsModelInputError("patch feature tensor contains NaN/Inf.")
                return output

        # Fallback: encode raw patch payload via patch encoder.
        encoded: Any = self._patch_encoder.encode(patches)
        encoded_tensor: torch.Tensor
        try:
            encoded_tensor = torch.as_tensor(encoded)
        except Exception as exc:  # noqa: BLE001
            raise ThreadsModelInputError(
                f"patch_encoder.encode output is not tensor-convertible: {exc}"
            ) from exc

        if encoded_tensor.ndim == 2:
            encoded_tensor = encoded_tensor.unsqueeze(0)
        elif encoded_tensor.ndim != 3:
            raise ThreadsModelInputError(
                "patch_encoder.encode output must be [N,F] or [B,N,F], "
                f"got shape={tuple(encoded_tensor.shape)}."
            )

        if int(encoded_tensor.shape[2]) != feature_dim:
            raise ThreadsModelInputError(
                "Encoded patch feature dim mismatch: "
                f"expected {feature_dim}, got {int(encoded_tensor.shape[2])}."
            )

        encoded_tensor = encoded_tensor.to(torch.float32)
        if self._validate_numerics and not torch.isfinite(encoded_tensor).all():
            raise ThreadsModelInputError("Encoded patch features contain NaN/Inf.")

        return encoded_tensor

    def _coerce_patch_mask(self, patch_mask: object, patch_features: torch.Tensor) -> torch.Tensor:
        """Coerce patch mask to bool tensor [B,N]."""
        try:
            mask_tensor: torch.Tensor = torch.as_tensor(patch_mask)
        except Exception as exc:  # noqa: BLE001
            raise ThreadsModelInputError(
                f"patch_mask cannot be converted to tensor: {exc}"
            ) from exc

        if mask_tensor.ndim == 1:
            mask_tensor = mask_tensor.unsqueeze(0)

        if mask_tensor.ndim != 2:
            raise ThreadsModelInputError(
                f"patch_mask must have shape [B,N], got {tuple(mask_tensor.shape)}."
            )

        expected_shape: Tuple[int, int] = (
            int(patch_features.shape[0]),
            int(patch_features.shape[1]),
        )
        actual_shape: Tuple[int, int] = (int(mask_tensor.shape[0]), int(mask_tensor.shape[1]))

        if actual_shape != expected_shape:
            raise ThreadsModelInputError(
                "patch_mask shape mismatch: "
                f"expected {expected_shape}, got {actual_shape}."
            )

        if mask_tensor.dtype != torch.bool:
            if torch.is_floating_point(mask_tensor):
                mask_tensor = mask_tensor > 0.5
            else:
                mask_tensor = mask_tensor > 0

        return mask_tensor

    def _validate_mask_has_valid_tokens(self, mask_tensor: torch.Tensor, shape_info: _ForwardShape) -> None:
        """Ensure each sample has at least one valid patch token."""
        valid_counts: torch.Tensor = mask_tensor.sum(dim=1)
        invalid_rows: torch.Tensor = torch.nonzero(valid_counts <= 0, as_tuple=False).flatten()
        if int(invalid_rows.numel()) > 0:
            invalid_indices: List[int] = [int(index) for index in invalid_rows.tolist()]
            raise ThreadsModelInputError(
                "Each sample must have at least one valid patch token. "
                f"invalid_rows={invalid_indices}; batch_shape="
                f"(B={shape_info.batch_size},N={shape_info.num_patches},F={shape_info.feature_dim})."
            )

    def _resolve_modality_indices(
        self,
        batch: Mapping[str, object],
        modality_flag_key: str,
        default_size: int,
    ) -> torch.Tensor:
        """Resolve available-sample indices for one modality."""
        if modality_flag_key in batch:
            flag_tensor: torch.Tensor = self._coerce_tensor(
                batch[modality_flag_key],
                name=modality_flag_key,
            )
            if flag_tensor.ndim != 1:
                raise ThreadsModelInputError(
                    f"{modality_flag_key} must be rank-1 [B], got shape={tuple(flag_tensor.shape)}."
                )
            if int(flag_tensor.shape[0]) != int(default_size):
                raise ThreadsModelInputError(
                    f"{modality_flag_key} length mismatch: expected {default_size}, got {int(flag_tensor.shape[0])}."
                )
            if flag_tensor.dtype != torch.bool:
                if torch.is_floating_point(flag_tensor):
                    flag_tensor = flag_tensor > 0.5
                else:
                    flag_tensor = flag_tensor > 0
            return torch.nonzero(flag_tensor, as_tuple=False).flatten().to(torch.long)

        # If explicit flag is missing, treat all rows as available.
        return torch.arange(int(default_size), dtype=torch.long)

    def _validate_dna_shape(self, dna_tensor: torch.Tensor) -> None:
        """Validate DNA tensor shape [B, 1673]."""
        if dna_tensor.ndim == 1:
            dna_tensor = dna_tensor.unsqueeze(0)

        if dna_tensor.ndim != 2:
            raise ThreadsModelInputError(
                f"{BATCH_KEY_DNA_MULTI_HOT} must be rank-2 [B,{self._dna_input_dim}], got {tuple(dna_tensor.shape)}."
            )
        if int(dna_tensor.shape[1]) != self._dna_input_dim:
            raise ThreadsModelInputError(
                f"{BATCH_KEY_DNA_MULTI_HOT} width mismatch: expected {self._dna_input_dim}, got {int(dna_tensor.shape[1])}."
            )

    def _weighted_mean_loss(self, loss_terms: Sequence[torch.Tensor], loss_weights: Sequence[float]) -> torch.Tensor:
        """Compute weighted mean over available modality losses."""
        if len(loss_terms) == 0:
            raise ThreadsModelRuntimeError("No loss terms available for aggregation.")
        if len(loss_terms) != len(loss_weights):
            raise ThreadsModelRuntimeError("Loss term count and weight count mismatch.")

        weight_tensor: torch.Tensor = torch.as_tensor(loss_weights, dtype=torch.float32, device=loss_terms[0].device)
        if (weight_tensor <= 0.0).any():
            raise ThreadsModelRuntimeError(
                f"All modality weights must be > 0, got {weight_tensor.tolist()}."
            )

        total_weight: torch.Tensor = weight_tensor.sum().clamp_min(DEFAULT_EPS)
        stacked_loss: torch.Tensor = torch.stack([term.to(torch.float32) for term in loss_terms], dim=0)
        weighted_sum: torch.Tensor = torch.sum(stacked_loss * weight_tensor)
        return weighted_sum / total_weight

    def _validate_loss_tensor(self, loss_tensor: torch.Tensor, name: str) -> None:
        """Validate scalar finite loss tensor."""
        if not isinstance(loss_tensor, torch.Tensor):
            raise ThreadsModelRuntimeError(f"{name} must be torch.Tensor.")

        if loss_tensor.ndim > 1:
            raise ThreadsModelRuntimeError(
                f"{name} must be scalar-like tensor, got shape={tuple(loss_tensor.shape)}."
            )

        if loss_tensor.ndim == 1 and int(loss_tensor.shape[0]) > 1:
            loss_tensor = loss_tensor.mean()

        if not torch.isfinite(loss_tensor):
            raise ThreadsModelRuntimeError(f"{name} is non-finite.")

    def _validate_first_dim(self, tensor_value: torch.Tensor, expected_batch: int, name: str) -> None:
        """Validate tensor first dimension equals batch size."""
        if tensor_value.ndim <= 0:
            raise ThreadsModelInputError(f"{name} must have batch dimension.")
        if int(tensor_value.shape[0]) != int(expected_batch):
            raise ThreadsModelInputError(
                f"{name} batch mismatch: expected {expected_batch}, got {int(tensor_value.shape[0])}."
            )

    def _coerce_tensor(self, value: object, name: str) -> torch.Tensor:
        """Coerce arbitrary value to tensor."""
        try:
            tensor_value: torch.Tensor = torch.as_tensor(value)
        except Exception as exc:  # noqa: BLE001
            raise ThreadsModelInputError(
                f"{name} cannot be converted to tensor: {exc}"
            ) from exc

        if self._validate_numerics and torch.is_floating_point(tensor_value):
            if not torch.isfinite(tensor_value).all():
                raise ThreadsModelInputError(f"{name} contains NaN/Inf values.")

        return tensor_value

    def _require_key(self, mapping: Mapping[str, object], key: str) -> object:
        """Get required key from mapping or raise input error."""
        if key not in mapping:
            raise ThreadsModelInputError(f"Missing required batch key: '{key}'.")
        return mapping[key]

    @staticmethod
    def _get_optional_int_attr(instance: object, attr_name: str) -> Optional[int]:
        """Read optional integer attribute from object if present."""
        if not hasattr(instance, attr_name):
            return None
        value: Any = getattr(instance, attr_name)
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        return None

    @staticmethod
    def _validate_patch_encoder(patch_encoder: PatchEncoder) -> PatchEncoder:
        if not isinstance(patch_encoder, PatchEncoder):
            raise ThreadsModelConfigError(
                f"patch_encoder must implement PatchEncoder, got {type(patch_encoder).__name__}."
            )
        return patch_encoder

    @staticmethod
    def _validate_slide_encoder(slide_encoder: ThreadsSlideEncoder) -> ThreadsSlideEncoder:
        if not isinstance(slide_encoder, ThreadsSlideEncoder):
            raise ThreadsModelConfigError(
                "slide_encoder must be ThreadsSlideEncoder, "
                f"got {type(slide_encoder).__name__}."
            )
        return slide_encoder

    @staticmethod
    def _validate_rna_encoder(rna_encoder: ScGPTRNAEncoder) -> ScGPTRNAEncoder:
        if not isinstance(rna_encoder, ScGPTRNAEncoder):
            raise ThreadsModelConfigError(
                f"rna_encoder must be ScGPTRNAEncoder, got {type(rna_encoder).__name__}."
            )
        return rna_encoder

    @staticmethod
    def _validate_dna_encoder(dna_encoder: DNAMLPEncoder) -> DNAMLPEncoder:
        if not isinstance(dna_encoder, DNAMLPEncoder):
            raise ThreadsModelConfigError(
                f"dna_encoder must be DNAMLPEncoder, got {type(dna_encoder).__name__}."
            )
        return dna_encoder

    @staticmethod
    def _validate_loss_fn(loss_fn: ContrastiveLoss) -> ContrastiveLoss:
        if not isinstance(loss_fn, ContrastiveLoss):
            raise ThreadsModelConfigError(
                f"loss_fn must be ContrastiveLoss, got {type(loss_fn).__name__}."
            )
        return loss_fn


__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_DNA_INPUT_DIM",
    "DEFAULT_MODALITY_WEIGHT_RNA",
    "DEFAULT_MODALITY_WEIGHT_DNA",
    "DEFAULT_VALIDATE_NUMERICS",
    "DEFAULT_SKIP_MISSING_MODALITY",
    "ThreadsModelError",
    "ThreadsModelConfigError",
    "ThreadsModelInputError",
    "ThreadsModelRuntimeError",
    "ThreadsModel",
]
