"""Typed experiment configuration schema and validation for TITAN reproduction.

This module centralizes configuration loading and validation for data processing,
training, and evaluation workflows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


# -----------------------------------------------------------------------------
# Paper-locked constants from provided config.yaml.
# -----------------------------------------------------------------------------
_PATCH_SIZE: int = 512
_MAGNIFICATION: str = "20x"
_FEATURE_DIM: int = 768
_STAGE1_REGION_PX: int = 8192
_STAGE1_REGION_GRID: Tuple[int, int] = (16, 16)
_STAGE3_GRID: Tuple[int, int] = (64, 64)
_STAGE3_REGION_PX: int = 32768

_EMBED_DIM: int = 768
_NUM_LAYERS: int = 6
_NUM_HEADS: int = 12
_HEAD_DIM: int = 64
_MLP_DIM: int = 3072

_ALLOWED_MODES: Tuple[str, ...] = (
    "prepare_data",
    "train_stage1",
    "train_stage2",
    "train_stage3",
    "eval",
)
_ALLOWED_STAGES: Tuple[str, ...] = (
    "stage1_titan_v",
    "stage2_roi_caption_alignment",
    "stage3_wsi_report_alignment",
)


class ConfigLoadError(ValueError):
    """Raised when config file loading fails."""


class ConfigValidationError(ValueError):
    """Raised when config validation fails."""


class DistributedConfig(BaseModel):
    """Distributed runtime settings."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    backend: str = "nccl"
    init_method: str = "env://"


class RuntimeConfig(BaseModel):
    """Runtime and reproducibility settings."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    num_workers: int = 8
    pin_memory: bool = True
    python_version: str = "3.9.16"
    pytorch_version: str = "2.0.1"
    cuda_version: str = "11.8"
    device: str = "cuda"
    distributed: DistributedConfig = Field(default_factory=DistributedConfig)

    @field_validator("seed", "num_workers")
    @classmethod
    def _non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Expected a non-negative integer.")
        return value


class HardwareStageConfig(BaseModel):
    """GPU settings for one pipeline stage."""

    model_config = ConfigDict(extra="forbid")

    gpus: int = 1
    gpu_type: str = "NVIDIA 3090 24GB"

    @field_validator("gpus")
    @classmethod
    def _validate_gpus(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("gpus must be > 0.")
        return value

    @field_validator("gpu_type")
    @classmethod
    def _validate_gpu_type(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("gpu_type must be a non-empty string.")
        return value


class HardwareConfig(BaseModel):
    """Hardware map by stage."""

    model_config = ConfigDict(extra="forbid")

    stage1: HardwareStageConfig = Field(
        default_factory=lambda: HardwareStageConfig(gpus=4, gpu_type="NVIDIA A100 80GB")
    )
    stage2: HardwareStageConfig = Field(
        default_factory=lambda: HardwareStageConfig(gpus=8, gpu_type="NVIDIA A100 80GB")
    )
    stage3: HardwareStageConfig = Field(
        default_factory=lambda: HardwareStageConfig(gpus=8, gpu_type="NVIDIA A100 80GB")
    )
    downstream_eval: HardwareStageConfig = Field(
        default_factory=lambda: HardwareStageConfig(gpus=1, gpu_type="NVIDIA 3090 24GB")
    )


class SegmentationConfig(BaseModel):
    """Tissue segmentation parameters."""

    model_config = ConfigDict(extra="forbid")

    hsv_saturation_threshold: int = 8
    median_blur_ksize: int = 7
    morph_close_ksize: int = 7
    min_contour_area: int = 256


class TissueGroupingConfig(BaseModel):
    """Tissue grouping parameters for stage-1 sampling."""

    model_config = ConfigDict(extra="forbid")

    method: str = "dbscan"
    min_patches: int = 16
    eps: Optional[float] = None
    min_samples: Optional[int] = None

    @field_validator("min_patches")
    @classmethod
    def _validate_min_patches(cls, value: int) -> int:
        if value < 1:
            raise ValueError("min_patches must be >= 1.")
        return value


class ManifestConfig(BaseModel):
    """Input metadata manifests."""

    model_config = ConfigDict(extra="forbid")

    wsi_manifest_csv: str = "./data/metadata/wsi_manifest.csv"
    roi_caption_pairs_jsonl: str = "./data/metadata/roi_caption_pairs.jsonl"
    wsi_report_pairs_jsonl: str = "./data/metadata/wsi_report_pairs.jsonl"
    splits_csv: str = "./data/metadata/splits.csv"


class DataConfig(BaseModel):
    """Data contract shared across pipeline components.

    Design-required fields:
    - wsi_root
    - meta_csv
    - patch_size
    - magnification
    - feature_dim
    - splits_path
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    wsi_root: str = "./data/wsi"
    meta_csv: str = "./data/metadata/wsi_manifest.csv"
    patch_size: int = Field(default=_PATCH_SIZE, alias="wsi_patch_size_px")
    magnification: str = _MAGNIFICATION
    feature_dim: int = Field(default=_FEATURE_DIM, alias="patch_feature_dim")
    splits_path: str = "./data/metadata/splits.csv"

    roi_region_size_px: int = _STAGE1_REGION_PX
    roi_region_grid_size: Tuple[int, int] = _STAGE1_REGION_GRID
    stage3_wsi_crop_grid_size: Tuple[int, int] = _STAGE3_GRID
    stage3_wsi_crop_size_px: int = _STAGE3_REGION_PX

    min_tissue_ratio: float = 0.5
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tissue_grouping: TissueGroupingConfig = Field(default_factory=TissueGroupingConfig)
    manifests: ManifestConfig = Field(default_factory=ManifestConfig)

    @field_validator("wsi_root", "meta_csv", "splits_path")
    @classmethod
    def _validate_non_empty_paths(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Path-like fields must be non-empty strings.")
        return value

    @model_validator(mode="after")
    def _validate_data_constants(self) -> "DataConfig":
        if self.patch_size != _PATCH_SIZE:
            raise ValueError(f"patch_size must be {_PATCH_SIZE}.")
        if self.magnification != _MAGNIFICATION:
            raise ValueError(f"magnification must be '{_MAGNIFICATION}'.")
        if self.feature_dim != _FEATURE_DIM:
            raise ValueError(f"feature_dim must be {_FEATURE_DIM}.")
        if tuple(self.roi_region_grid_size) != _STAGE1_REGION_GRID:
            raise ValueError(f"roi_region_grid_size must be {_STAGE1_REGION_GRID}.")
        if self.roi_region_size_px != _STAGE1_REGION_PX:
            raise ValueError(f"roi_region_size_px must be {_STAGE1_REGION_PX}.")
        if tuple(self.stage3_wsi_crop_grid_size) != _STAGE3_GRID:
            raise ValueError(f"stage3_wsi_crop_grid_size must be {_STAGE3_GRID}.")
        if self.stage3_wsi_crop_size_px != _STAGE3_REGION_PX:
            raise ValueError(f"stage3_wsi_crop_size_px must be {_STAGE3_REGION_PX}.")

        if self.roi_region_size_px != self.patch_size * self.roi_region_grid_size[0]:
            raise ValueError("roi_region_size_px must equal patch_size * roi_region_grid_size[0].")
        if self.stage3_wsi_crop_size_px != self.patch_size * self.stage3_wsi_crop_grid_size[0]:
            raise ValueError(
                "stage3_wsi_crop_size_px must equal patch_size * stage3_wsi_crop_grid_size[0]."
            )
        return self


class ModelConfig(BaseModel):
    """Slide encoder contract.

    Design-required fields:
    - embed_dim
    - num_layers
    - num_heads
    - mlp_dim
    - use_alibi_2d
    - max_tokens_train
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    embed_dim: int = Field(default=_EMBED_DIM, alias="embedding_dim")
    num_layers: int = Field(default=_NUM_LAYERS, alias="num_attention_layers")
    num_heads: int = Field(default=_NUM_HEADS, alias="num_attention_heads")
    mlp_dim: int = Field(default=_MLP_DIM, alias="mlp_hidden_dim")
    use_alibi_2d: bool = True
    max_tokens_train: int = 256

    head_dim: int = _HEAD_DIM
    positional_encoding: str = "2D ALiBi (Euclidean-distance based bias)"
    architecture: str = "ViT in feature space"

    @model_validator(mode="after")
    def _validate_model_constants(self) -> "ModelConfig":
        if self.embed_dim != _EMBED_DIM:
            raise ValueError(f"embed_dim must be {_EMBED_DIM}.")
        if self.num_layers != _NUM_LAYERS:
            raise ValueError(f"num_layers must be {_NUM_LAYERS}.")
        if self.num_heads != _NUM_HEADS:
            raise ValueError(f"num_heads must be {_NUM_HEADS}.")
        if self.mlp_dim != _MLP_DIM:
            raise ValueError(f"mlp_dim must be {_MLP_DIM}.")
        if self.head_dim != _HEAD_DIM:
            raise ValueError(f"head_dim must be {_HEAD_DIM}.")
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError("num_heads * head_dim must equal embed_dim.")
        if not self.use_alibi_2d:
            raise ValueError("use_alibi_2d must be True for TITAN configuration.")
        if self.max_tokens_train <= 0:
            raise ValueError("max_tokens_train must be > 0.")
        return self


class MultimodalConfig(BaseModel):
    """Multimodal CoCa configuration contract."""

    model_config = ConfigDict(extra="forbid")

    framework: str = "CoCa"
    reconstruction_queries: int = 128
    text_encoder_source: str = "CONCHv1.5 pretrained text encoder"
    text_decoder_source: str = "CONCHv1.5 pretrained multimodal decoder"
    text_encoder_layers: int = 12
    text_decoder_layers: int = 12
    text_embedding_dim: int = _EMBED_DIM
    text_hidden_dim: int = _MLP_DIM

    @model_validator(mode="after")
    def _validate_multimodal_constants(self) -> "MultimodalConfig":
        if self.framework != "CoCa":
            raise ValueError("framework must be 'CoCa'.")
        if self.reconstruction_queries != 128:
            raise ValueError("reconstruction_queries must be 128.")
        if self.text_encoder_layers != 12:
            raise ValueError("text_encoder_layers must be 12.")
        if self.text_decoder_layers != 12:
            raise ValueError("text_decoder_layers must be 12.")
        if self.text_embedding_dim != _EMBED_DIM:
            raise ValueError(f"text_embedding_dim must be {_EMBED_DIM}.")
        if self.text_hidden_dim != _MLP_DIM:
            raise ValueError(f"text_hidden_dim must be {_MLP_DIM}.")
        return self


class SlideEncoderContainerConfig(BaseModel):
    """Wrapper for slide encoder section in full config."""

    model_config = ConfigDict(extra="forbid")

    architecture: str = "ViT in feature space"
    num_layers: int = _NUM_LAYERS
    num_attention_heads: int = _NUM_HEADS
    head_dim: int = _HEAD_DIM
    embedding_dim: int = _EMBED_DIM
    mlp_hidden_dim: int = _MLP_DIM
    positional_encoding: str = "2D ALiBi (Euclidean-distance based bias)"
    use_alibi_2d: bool = True
    alibi_slopes: Optional[List[float]] = None


class ModelBundleConfig(BaseModel):
    """Top-level model section."""

    model_config = ConfigDict(extra="forbid")

    slide_encoder: SlideEncoderContainerConfig = Field(default_factory=SlideEncoderContainerConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)


class OptimizerConfig(BaseModel):
    """Optimizer settings with nullable paper-missing parameters."""

    model_config = ConfigDict(extra="forbid")

    name: str = "adamw"
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    scheduler: Optional[str] = None
    warmup: Optional[int] = None


class Stage1ViewSamplingConfig(BaseModel):
    """Stage-1 iBOT view policy."""

    model_config = ConfigDict(extra="forbid")

    global_views: int = 2
    global_view_grid_size: Tuple[int, int] = (14, 14)
    local_views: int = 10
    local_view_grid_size: Tuple[int, int] = (6, 6)


class Stage1IBOTConfig(BaseModel):
    """Stage-1 iBOT unresolved supplementary parameters."""

    model_config = ConfigDict(extra="forbid")

    student_temperature: Optional[float] = None
    teacher_temperature: Optional[float] = None
    center_momentum: Optional[float] = None
    mask_ratio: Optional[float] = None
    ema_momentum: Optional[float] = None


class StageLossConfig(BaseModel):
    """Stage 2/3 multimodal loss weights."""

    model_config = ConfigDict(extra="forbid")

    contrastive_weight: Optional[float] = None
    caption_weight: Optional[float] = None


class TrainConfig(BaseModel):
    """Generic train contract.

    Design-required fields:
    - stage
    - epochs
    - lr
    - weight_decay
    - batch_size
    - grad_accum_steps
    - mixed_precision
    """

    model_config = ConfigDict(extra="forbid")

    stage: str = "stage1_titan_v"
    epochs: int = 1
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    batch_size: int = 1
    grad_accum_steps: int = 1
    mixed_precision: bool = True

    objective: str = ""
    effective_batch_size: int = 1

    @field_validator("epochs", "batch_size", "grad_accum_steps", "effective_batch_size")
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Expected a positive integer.")
        return value


class Stage1TrainConfig(TrainConfig):
    """Stage-1 configuration."""

    stage: Literal["stage1_titan_v"] = "stage1_titan_v"
    epochs: int = 270
    batch_size: int = 256
    grad_accum_steps: int = 1
    effective_batch_size: int = 1024
    objective: str = "iBOT (student-teacher distillation + masked image modeling) in feature space"

    iterations: int = 91260
    view_sampling: Stage1ViewSamplingConfig = Field(default_factory=Stage1ViewSamplingConfig)
    augmentations: List[str] = Field(
        default_factory=lambda: ["horizontal_flip", "vertical_flip", "feature_posterization"]
    )
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    ibot: Stage1IBOTConfig = Field(default_factory=Stage1IBOTConfig)

    @model_validator(mode="after")
    def _validate_stage1(self) -> "Stage1TrainConfig":
        if self.iterations != 91260:
            raise ValueError("Stage1 iterations must be 91260.")
        required = {"horizontal_flip", "vertical_flip", "feature_posterization"}
        if set(self.augmentations) != required:
            raise ValueError("Stage1 augmentations must exactly match required feature-space augmentations.")
        if self.view_sampling.global_views != 2:
            raise ValueError("Stage1 global_views must be 2.")
        if tuple(self.view_sampling.global_view_grid_size) != (14, 14):
            raise ValueError("Stage1 global_view_grid_size must be (14, 14).")
        if self.view_sampling.local_views != 10:
            raise ValueError("Stage1 local_views must be 10.")
        if tuple(self.view_sampling.local_view_grid_size) != (6, 6):
            raise ValueError("Stage1 local_view_grid_size must be (6, 6).")
        return self


class Stage2TrainConfig(TrainConfig):
    """Stage-2 configuration."""

    stage: Literal["stage2_roi_caption_alignment"] = "stage2_roi_caption_alignment"
    epochs: int = 1
    batch_size: int = 196
    grad_accum_steps: int = 2
    effective_batch_size: int = 3136
    objective: str = "CoCa contrastive + generative alignment on ROI-caption pairs"

    num_pairs: int = 423122
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    losses: StageLossConfig = Field(default_factory=StageLossConfig)

    @model_validator(mode="after")
    def _validate_stage2(self) -> "Stage2TrainConfig":
        if self.num_pairs != 423122:
            raise ValueError("Stage2 num_pairs must be 423122.")
        return self


class Stage3OptimizerConfig(OptimizerConfig):
    """Stage-3 optimizer extension with lower-LR backbone options."""

    vision_backbone_learning_rate: Optional[float] = None
    vision_backbone_weight_decay: Optional[float] = None


class Stage3TrainConfig(TrainConfig):
    """Stage-3 configuration."""

    stage: Literal["stage3_wsi_report_alignment"] = "stage3_wsi_report_alignment"
    epochs: int = 1
    batch_size: int = 16
    grad_accum_steps: int = 2
    effective_batch_size: int = 256
    objective: str = "CoCa contrastive + generative alignment on WSI-report pairs"

    num_pairs: int = 182862
    notes: List[str] = Field(
        default_factory=lambda: [
            "Use smaller learning rate and weight decay for vision backbone",
            "Use slow warm-up for vision backbone",
        ]
    )
    optimizer: Stage3OptimizerConfig = Field(default_factory=Stage3OptimizerConfig)
    losses: StageLossConfig = Field(default_factory=StageLossConfig)

    @model_validator(mode="after")
    def _validate_stage3(self) -> "Stage3TrainConfig":
        if self.num_pairs != 182862:
            raise ValueError("Stage3 num_pairs must be 182862.")
        if not self.notes:
            raise ValueError("Stage3 notes cannot be empty.")
        return self


class PrecisionConfig(BaseModel):
    """AMP precision options."""

    model_config = ConfigDict(extra="forbid")

    use_amp: bool = True
    amp_dtype: str = "float16"


class TrainingConfig(BaseModel):
    """Training section with stage sub-profiles."""

    model_config = ConfigDict(extra="forbid")

    precision: PrecisionConfig = Field(default_factory=PrecisionConfig)
    grad_clip_norm: Optional[float] = None
    log_every_n_steps: int = 20
    save_every_n_steps: int = 1000
    val_every_n_steps: Optional[int] = None

    stage1_titan_v: Stage1TrainConfig = Field(default_factory=Stage1TrainConfig)
    stage2_roi_caption_alignment: Stage2TrainConfig = Field(default_factory=Stage2TrainConfig)
    stage3_wsi_report_alignment: Stage3TrainConfig = Field(default_factory=Stage3TrainConfig)


class LinearProbeDefaultsConfig(BaseModel):
    """Fallback linear probe settings for no-validation settings."""

    model_config = ConfigDict(extra="forbid")

    l2: float = 1.0
    max_iter: int = 1000


class L2GridConfig(BaseModel):
    """Linear-probe L2 regularization grid."""

    model_config = ConfigDict(extra="forbid")

    count: int = 45
    min: float = 1.0e-6
    max: float = 10.0
    spacing: str = "log"

    @model_validator(mode="after")
    def _validate_l2_grid(self) -> "L2GridConfig":
        if self.count != 45:
            raise ValueError("linear_probe.l2_grid.count must be 45.")
        if abs(self.min - 1.0e-6) > 1e-15:
            raise ValueError("linear_probe.l2_grid.min must be 1e-6.")
        if abs(self.max - 10.0) > 1e-15:
            raise ValueError("linear_probe.l2_grid.max must be 10.")
        if self.spacing != "log":
            raise ValueError("linear_probe.l2_grid.spacing must be 'log'.")
        return self


class LinearProbeEvalConfig(BaseModel):
    """Linear probe evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    method: str = "scikit-learn logistic regression (L-BFGS)"
    l2_grid: L2GridConfig = Field(default_factory=L2GridConfig)
    max_iter: int = 500
    few_shot_or_no_val_defaults: LinearProbeDefaultsConfig = Field(
        default_factory=LinearProbeDefaultsConfig
    )

    @model_validator(mode="after")
    def _validate_linear_probe(self) -> "LinearProbeEvalConfig":
        if self.max_iter != 500:
            raise ValueError("linear_probe.max_iter must be 500.")
        return self


class KNNProbeEvalConfig(BaseModel):
    """k-NN probing settings."""

    model_config = ConfigDict(extra="forbid")

    k: int = 20
    distance: str = "euclidean"
    preprocess: List[str] = Field(default_factory=lambda: ["center", "l2_normalize"])

    @model_validator(mode="after")
    def _validate_knn(self) -> "KNNProbeEvalConfig":
        if self.k != 20:
            raise ValueError("knn_probe.k must be 20.")
        if self.distance != "euclidean":
            raise ValueError("knn_probe.distance must be 'euclidean'.")
        required = {"center", "l2_normalize"}
        if not required.issubset(set(self.preprocess)):
            raise ValueError("knn_probe.preprocess must include 'center' and 'l2_normalize'.")
        return self


class FewShotEvalConfig(BaseModel):
    """Few-shot protocol settings."""

    model_config = ConfigDict(extra="forbid")

    shots: List[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    runs: int = 50

    @model_validator(mode="after")
    def _validate_few_shot(self) -> "FewShotEvalConfig":
        if self.shots != [1, 2, 4, 8, 16, 32]:
            raise ValueError("few_shot.shots must be [1,2,4,8,16,32].")
        if self.runs != 50:
            raise ValueError("few_shot.runs must be 50.")
        return self


class RetrievalEvalConfig(BaseModel):
    """Retrieval protocol settings."""

    model_config = ConfigDict(extra="forbid")

    slide_retrieval_k: List[int] = Field(default_factory=lambda: [1, 3, 5])
    cross_modal_recall_k: List[int] = Field(default_factory=lambda: [1, 3, 5, 10])

    @model_validator(mode="after")
    def _validate_retrieval(self) -> "RetrievalEvalConfig":
        if self.slide_retrieval_k != [1, 3, 5]:
            raise ValueError("retrieval.slide_retrieval_k must be [1,3,5].")
        if self.cross_modal_recall_k != [1, 3, 5, 10]:
            raise ValueError("retrieval.cross_modal_recall_k must be [1,3,5,10].")
        return self


class ReportGenerationEvalConfig(BaseModel):
    """Report generation settings."""

    model_config = ConfigDict(extra="forbid")

    decoding: str = "beam_search"
    num_beams: int = 5
    num_beam_groups: int = 1

    @model_validator(mode="after")
    def _validate_generation(self) -> "ReportGenerationEvalConfig":
        if self.decoding != "beam_search":
            raise ValueError("report_generation.decoding must be 'beam_search'.")
        if self.num_beams != 5:
            raise ValueError("report_generation.num_beams must be 5.")
        if self.num_beam_groups != 1:
            raise ValueError("report_generation.num_beam_groups must be 1.")
        return self


class SurvivalEvalConfig(BaseModel):
    """Survival evaluation settings."""

    model_config = ConfigDict(extra="forbid")

    model: str = "linear Cox proportional hazards"
    package: str = "scikit-survival"

    @model_validator(mode="after")
    def _validate_survival(self) -> "SurvivalEvalConfig":
        if self.model != "linear Cox proportional hazards":
            raise ValueError("survival.model must be 'linear Cox proportional hazards'.")
        if self.package != "scikit-survival":
            raise ValueError("survival.package must be 'scikit-survival'.")
        return self


class EvaluationConfig(BaseModel):
    """Evaluation configuration section."""

    model_config = ConfigDict(extra="forbid")

    bootstrap_samples: int = 1000
    linear_probe: LinearProbeEvalConfig = Field(default_factory=LinearProbeEvalConfig)
    knn_probe: KNNProbeEvalConfig = Field(default_factory=KNNProbeEvalConfig)
    few_shot: FewShotEvalConfig = Field(default_factory=FewShotEvalConfig)
    retrieval: RetrievalEvalConfig = Field(default_factory=RetrievalEvalConfig)
    report_generation: ReportGenerationEvalConfig = Field(default_factory=ReportGenerationEvalConfig)
    survival: SurvivalEvalConfig = Field(default_factory=SurvivalEvalConfig)


class PathsConfig(BaseModel):
    """Common project paths."""

    model_config = ConfigDict(extra="forbid")

    project_root: str = "."
    data_root: str = "./data"
    output_root: str = "./outputs"
    logs_root: str = "./outputs/logs"
    checkpoints_root: str = "./outputs/checkpoints"
    artifacts_root: str = "./outputs/artifacts"


class ArtifactsConfig(BaseModel):
    """Artifact naming contract."""

    model_config = ConfigDict(extra="forbid")

    features_h5_name: str = "features.h5"
    feature_grid_name: str = "grid.pt"
    pairs_jsonl_name: str = "pairs.jsonl"
    splits_csv_name: str = "splits.csv"
    tissue_groups_name: str = "groups.json"
    stage1_checkpoint_name: str = "titan_v.ckpt"
    stage2_checkpoint_name: str = "titan_stage2.ckpt"
    stage3_checkpoint_name: str = "titan_final.ckpt"


class InitCheckpointsConfig(BaseModel):
    """Optional initialization checkpoints per stage."""

    model_config = ConfigDict(extra="forbid")

    stage1_from: Optional[str] = None
    stage2_from: Optional[str] = None
    stage3_from: Optional[str] = None
    eval_from: Optional[str] = None


class LoggingConfig(BaseModel):
    """Logging outputs."""

    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"
    save_json: bool = True
    save_csv: bool = True
    save_tensorboard: bool = False


class NotesConfig(BaseModel):
    """Human-readable notes."""

    model_config = ConfigDict(extra="forbid")

    source_of_truth: List[str] = Field(default_factory=list)


class ExperimentConfig(BaseModel):
    """Top-level typed experiment config with loader and validation."""

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    mode: str = "prepare_data"
    stage: str = "stage1_titan_v"

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    artifacts: ArtifactsConfig = Field(default_factory=ArtifactsConfig)
    init_checkpoints: InitCheckpointsConfig = Field(default_factory=InitCheckpointsConfig)

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelBundleConfig = Field(default_factory=ModelBundleConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    notes: NotesConfig = Field(default_factory=NotesConfig)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        if value not in _ALLOWED_MODES:
            raise ValueError(f"Unsupported mode: {value!r}. Allowed values: {_ALLOWED_MODES}")
        return value

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, value: str) -> str:
        if value not in _ALLOWED_STAGES:
            raise ValueError(f"Unsupported stage: {value!r}. Allowed values: {_ALLOWED_STAGES}")
        return value

    @model_validator(mode="after")
    def _validate_cross_section_contracts(self) -> "ExperimentConfig":
        model_cfg = ModelConfig(
            embedding_dim=self.model.slide_encoder.embedding_dim,
            num_attention_layers=self.model.slide_encoder.num_layers,
            num_attention_heads=self.model.slide_encoder.num_attention_heads,
            mlp_hidden_dim=self.model.slide_encoder.mlp_hidden_dim,
            use_alibi_2d=self.model.slide_encoder.use_alibi_2d,
            max_tokens_train=256,
            head_dim=self.model.slide_encoder.head_dim,
            positional_encoding=self.model.slide_encoder.positional_encoding,
            architecture=self.model.slide_encoder.architecture,
        )

        if self.data.feature_dim != model_cfg.embed_dim:
            raise ValueError("data.feature_dim must equal model.slide_encoder.embedding_dim.")
        if self.model.multimodal.text_embedding_dim != model_cfg.embed_dim:
            raise ValueError("model.multimodal.text_embedding_dim must equal slide embedding dim.")

        s1 = self.training.stage1_titan_v
        s2 = self.training.stage2_roi_caption_alignment
        s3 = self.training.stage3_wsi_report_alignment

        expected_s1 = s1.batch_size * self.hardware.stage1.gpus * s1.grad_accum_steps
        if expected_s1 != s1.effective_batch_size:
            raise ValueError("Stage1 effective_batch_size mismatch with hardware and grad accumulation.")

        expected_s2 = s2.batch_size * self.hardware.stage2.gpus * s2.grad_accum_steps
        if expected_s2 != s2.effective_batch_size:
            raise ValueError("Stage2 effective_batch_size mismatch with hardware and grad accumulation.")

        expected_s3 = s3.batch_size * self.hardware.stage3.gpus * s3.grad_accum_steps
        if expected_s3 != s3.effective_batch_size:
            raise ValueError("Stage3 effective_batch_size mismatch with hardware and grad accumulation.")

        if self.data.roi_region_grid_size[0] * self.data.roi_region_grid_size[1] != 256:
            raise ValueError("Stage1 region token count must be 256 (16x16).")
        if self.data.stage3_wsi_crop_grid_size[0] * self.data.stage3_wsi_crop_grid_size[1] != 4096:
            raise ValueError("Stage3 crop token count must be 4096 (64x64).")

        return self

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load and validate an ExperimentConfig from a YAML file."""
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists() or not config_path.is_file():
            raise ConfigLoadError(f"Config file not found: {config_path}")

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except OSError as exc:
            raise ConfigLoadError(f"Failed reading config file: {config_path}") from exc
        except yaml.YAMLError as exc:
            raise ConfigLoadError(f"Invalid YAML format: {config_path}") from exc

        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ConfigLoadError("Top-level YAML structure must be a mapping.")

        normalized = _normalize_to_canonical(raw)
        try:
            cfg = cls.model_validate(normalized)
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc

        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Run explicit validation and raise ConfigValidationError on failure."""
        try:
            self.__class__.model_validate(self.model_dump())
        except ValidationError as exc:
            raise ConfigValidationError(str(exc)) from exc


def _canonical_defaults() -> Dict[str, Any]:
    """Return canonical defaults from provided config.yaml constants."""
    return {
        "version": 1,
        "mode": "prepare_data",
        "stage": "stage1_titan_v",
        "runtime": {
            "seed": 42,
            "deterministic": True,
            "benchmark": False,
            "num_workers": 8,
            "pin_memory": True,
            "python_version": "3.9.16",
            "pytorch_version": "2.0.1",
            "cuda_version": "11.8",
            "device": "cuda",
            "distributed": {"enabled": False, "backend": "nccl", "init_method": "env://"},
        },
        "hardware": {
            "stage1": {"gpus": 4, "gpu_type": "NVIDIA A100 80GB"},
            "stage2": {"gpus": 8, "gpu_type": "NVIDIA A100 80GB"},
            "stage3": {"gpus": 8, "gpu_type": "NVIDIA A100 80GB"},
            "downstream_eval": {"gpus": 1, "gpu_type": "NVIDIA 3090 24GB"},
        },
        "paths": {
            "project_root": ".",
            "data_root": "./data",
            "output_root": "./outputs",
            "logs_root": "./outputs/logs",
            "checkpoints_root": "./outputs/checkpoints",
            "artifacts_root": "./outputs/artifacts",
        },
        "artifacts": {
            "features_h5_name": "features.h5",
            "feature_grid_name": "grid.pt",
            "pairs_jsonl_name": "pairs.jsonl",
            "splits_csv_name": "splits.csv",
            "tissue_groups_name": "groups.json",
            "stage1_checkpoint_name": "titan_v.ckpt",
            "stage2_checkpoint_name": "titan_stage2.ckpt",
            "stage3_checkpoint_name": "titan_final.ckpt",
        },
        "init_checkpoints": {
            "stage1_from": None,
            "stage2_from": None,
            "stage3_from": None,
            "eval_from": None,
        },
        "data": {
            "wsi_root": "./data/wsi",
            "meta_csv": "./data/metadata/wsi_manifest.csv",
            "wsi_patch_size_px": 512,
            "magnification": "20x",
            "patch_feature_dim": 768,
            "splits_path": "./data/metadata/splits.csv",
            "roi_region_size_px": 8192,
            "roi_region_grid_size": [16, 16],
            "stage3_wsi_crop_grid_size": [64, 64],
            "stage3_wsi_crop_size_px": 32768,
            "min_tissue_ratio": 0.5,
            "segmentation": {
                "hsv_saturation_threshold": 8,
                "median_blur_ksize": 7,
                "morph_close_ksize": 7,
                "min_contour_area": 256,
            },
            "tissue_grouping": {"method": "dbscan", "min_patches": 16, "eps": None, "min_samples": None},
            "manifests": {
                "wsi_manifest_csv": "./data/metadata/wsi_manifest.csv",
                "roi_caption_pairs_jsonl": "./data/metadata/roi_caption_pairs.jsonl",
                "wsi_report_pairs_jsonl": "./data/metadata/wsi_report_pairs.jsonl",
                "splits_csv": "./data/metadata/splits.csv",
            },
        },
        "model": {
            "slide_encoder": {
                "architecture": "ViT in feature space",
                "num_layers": 6,
                "num_attention_heads": 12,
                "head_dim": 64,
                "embedding_dim": 768,
                "mlp_hidden_dim": 3072,
                "positional_encoding": "2D ALiBi (Euclidean-distance based bias)",
                "use_alibi_2d": True,
                "alibi_slopes": None,
            },
            "multimodal": {
                "framework": "CoCa",
                "reconstruction_queries": 128,
                "text_encoder_source": "CONCHv1.5 pretrained text encoder",
                "text_decoder_source": "CONCHv1.5 pretrained multimodal decoder",
                "text_encoder_layers": 12,
                "text_decoder_layers": 12,
                "text_embedding_dim": 768,
                "text_hidden_dim": 3072,
            },
        },
        "training": {
            "precision": {"use_amp": True, "amp_dtype": "float16"},
            "grad_clip_norm": None,
            "log_every_n_steps": 20,
            "save_every_n_steps": 1000,
            "val_every_n_steps": None,
            "stage1_titan_v": {
                "stage": "stage1_titan_v",
                "epochs": 270,
                "batch_size": 256,
                "grad_accum_steps": 1,
                "effective_batch_size": 1024,
                "objective": "iBOT (student-teacher distillation + masked image modeling) in feature space",
                "iterations": 91260,
                "view_sampling": {
                    "global_views": 2,
                    "global_view_grid_size": [14, 14],
                    "local_views": 10,
                    "local_view_grid_size": [6, 6],
                },
                "augmentations": ["horizontal_flip", "vertical_flip", "feature_posterization"],
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": None,
                    "weight_decay": None,
                    "betas": None,
                    "scheduler": None,
                    "warmup": None,
                },
                "ibot": {
                    "student_temperature": None,
                    "teacher_temperature": None,
                    "center_momentum": None,
                    "mask_ratio": None,
                    "ema_momentum": None,
                },
            },
            "stage2_roi_caption_alignment": {
                "stage": "stage2_roi_caption_alignment",
                "epochs": 1,
                "batch_size": 196,
                "grad_accum_steps": 2,
                "effective_batch_size": 3136,
                "objective": "CoCa contrastive + generative alignment on ROI-caption pairs",
                "num_pairs": 423122,
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": None,
                    "weight_decay": None,
                    "betas": None,
                    "scheduler": None,
                    "warmup": None,
                },
                "losses": {"contrastive_weight": None, "caption_weight": None},
            },
            "stage3_wsi_report_alignment": {
                "stage": "stage3_wsi_report_alignment",
                "epochs": 1,
                "batch_size": 16,
                "grad_accum_steps": 2,
                "effective_batch_size": 256,
                "objective": "CoCa contrastive + generative alignment on WSI-report pairs",
                "num_pairs": 182862,
                "notes": [
                    "Use smaller learning rate and weight decay for vision backbone",
                    "Use slow warm-up for vision backbone",
                ],
                "optimizer": {
                    "name": "adamw",
                    "learning_rate": None,
                    "weight_decay": None,
                    "betas": None,
                    "scheduler": None,
                    "warmup": None,
                    "vision_backbone_learning_rate": None,
                    "vision_backbone_weight_decay": None,
                },
                "losses": {"contrastive_weight": None, "caption_weight": None},
            },
        },
        "evaluation": {
            "bootstrap_samples": 1000,
            "linear_probe": {
                "method": "scikit-learn logistic regression (L-BFGS)",
                "l2_grid": {"count": 45, "min": 1.0e-6, "max": 10.0, "spacing": "log"},
                "max_iter": 500,
                "few_shot_or_no_val_defaults": {"l2": 1.0, "max_iter": 1000},
            },
            "knn_probe": {"k": 20, "distance": "euclidean", "preprocess": ["center", "l2_normalize"]},
            "few_shot": {"shots": [1, 2, 4, 8, 16, 32], "runs": 50},
            "retrieval": {"slide_retrieval_k": [1, 3, 5], "cross_modal_recall_k": [1, 3, 5, 10]},
            "report_generation": {"decoding": "beam_search", "num_beams": 5, "num_beam_groups": 1},
            "survival": {"model": "linear Cox proportional hazards", "package": "scikit-survival"},
        },
        "logging": {"level": "INFO", "save_json": True, "save_csv": True, "save_tensorboard": False},
        "notes": {
            "source_of_truth": [
                "Hyperparameters not listed above are not provided in the supplied paper text and are intentionally set to null.",
                "Supplementary tables are required to fill null optimizer/scheduler/loss settings.",
            ]
        },
    }


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base, mutating and returning base."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_to_canonical(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize different config file shapes into canonical ExperimentConfig schema."""
    canonical = _canonical_defaults()

    if "runtime" in raw and "hardware" in raw and "data" in raw and "model" in raw:
        return _deep_merge(canonical, raw)

    normalized: Dict[str, Any] = {}

    if "train" in raw and isinstance(raw["train"], dict):
        train_dict = raw["train"]
        stage_token = str(train_dict.get("stage", "stage1")).strip().lower()
        stage_map = {
            "stage1": "stage1_titan_v",
            "stage1_titan_v": "stage1_titan_v",
            "stage2": "stage2_roi_caption_alignment",
            "stage2_roi_caption_alignment": "stage2_roi_caption_alignment",
            "stage3": "stage3_wsi_report_alignment",
            "stage3_wsi_report_alignment": "stage3_wsi_report_alignment",
        }
        resolved_stage = stage_map.get(stage_token, "stage1_titan_v")
        normalized["mode"] = {
            "stage1_titan_v": "train_stage1",
            "stage2_roi_caption_alignment": "train_stage2",
            "stage3_wsi_report_alignment": "train_stage3",
        }[resolved_stage]
        normalized["stage"] = resolved_stage

        if "runtime" in train_dict:
            normalized["runtime"] = train_dict["runtime"]
        if "data" in train_dict:
            normalized["data"] = _normalize_data_short(train_dict["data"])
        if "model" in train_dict:
            normalized["model"] = _normalize_model_short(train_dict["model"])

        training_patch = _normalize_training_short(train_dict, resolved_stage)
        normalized["training"] = training_patch

    if "eval" in raw and isinstance(raw["eval"], dict):
        eval_dict = raw["eval"]
        normalized["mode"] = "eval"
        if "runtime" in eval_dict:
            normalized["runtime"] = eval_dict["runtime"]
        if "data" in eval_dict:
            normalized["data"] = _normalize_data_short(eval_dict["data"])

        eval_name = str(eval_dict.get("name", "")).strip()
        evaluation_patch: Dict[str, Any] = {}
        if eval_name == "linear_probe":
            evaluation_patch["linear_probe"] = {
                "method": eval_dict.get("method", "scikit-learn logistic regression (L-BFGS)"),
                "l2_grid": eval_dict.get("l2_grid", {}),
                "max_iter": eval_dict.get("max_iter", 500),
                "few_shot_or_no_val_defaults": eval_dict.get("few_shot_or_no_val_defaults", {}),
            }
        elif eval_name == "few_shot":
            evaluation_patch["few_shot"] = eval_dict.get("protocol", {})
        elif eval_name == "retrieval":
            protocol = eval_dict.get("protocol", {})
            evaluation_patch["retrieval"] = {
                "slide_retrieval_k": protocol.get("unimodal_slide_retrieval", {}).get("k_values", [1, 3, 5]),
                "cross_modal_recall_k": protocol.get("cross_modal_retrieval", {}).get("recall_k_values", [1, 3, 5, 10]),
            }
        elif eval_name == "report_generation":
            decoding = eval_dict.get("protocol", {}).get("decoding", {})
            evaluation_patch["report_generation"] = {
                "decoding": decoding.get("strategy", "beam_search"),
                "num_beams": decoding.get("num_beams", 5),
                "num_beam_groups": decoding.get("num_beam_groups", 1),
            }
        elif eval_name == "survival":
            evaluation_patch["survival"] = eval_dict.get("protocol", {})
        elif eval_name == "zero_shot":
            # Zero-shot has no dedicated subsection in canonical evaluation block;
            # its settings are covered by retrieval/report/linear defaults.
            pass

        if evaluation_patch:
            normalized["evaluation"] = evaluation_patch

    if "wsi_root" in raw or "patch_size" in raw or "patch_feature_dim" in raw:
        normalized["mode"] = raw.get("mode", "prepare_data")
        normalized["data"] = _normalize_data_short(raw)

    return _deep_merge(canonical, normalized)


def _normalize_data_short(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize short-form data keys into canonical DataConfig keys."""
    out = dict(data_dict)

    if "patch_size" in out and "wsi_patch_size_px" not in out:
        out["wsi_patch_size_px"] = out.pop("patch_size")
    if "feature_dim" in out and "patch_feature_dim" not in out:
        out["patch_feature_dim"] = out.pop("feature_dim")

    if "meta_csv" not in out:
        manifest = out.get("manifests", {})
        if isinstance(manifest, dict) and "wsi_manifest_csv" in manifest:
            out["meta_csv"] = manifest["wsi_manifest_csv"]

    if "splits_path" not in out:
        manifest = out.get("manifests", {})
        if isinstance(manifest, dict) and "splits_csv" in manifest:
            out["splits_path"] = manifest["splits_csv"]

    return out


def _normalize_model_short(model_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize stage-file model structure to canonical model section."""
    out: Dict[str, Any] = {}

    if "slide_encoder" in model_dict or "multimodal" in model_dict:
        return model_dict

    if "encoder" in model_dict:
        encoder = dict(model_dict.get("encoder", {}))
        out["slide_encoder"] = {
            "architecture": model_dict.get("architecture", "ViT in feature space"),
            "num_layers": encoder.get("num_layers", _NUM_LAYERS),
            "num_attention_heads": encoder.get("num_heads", _NUM_HEADS),
            "head_dim": encoder.get("head_dim", _HEAD_DIM),
            "embedding_dim": encoder.get("embed_dim", _EMBED_DIM),
            "mlp_hidden_dim": encoder.get("mlp_dim", _MLP_DIM),
            "positional_encoding": encoder.get(
                "positional_encoding", "2D ALiBi (Euclidean-distance based bias)"
            ),
            "use_alibi_2d": encoder.get("use_alibi_2d", True),
            "alibi_slopes": model_dict.get("alibi_2d", {}).get("slopes"),
        }

    if "text_encoder" in model_dict or "poolers" in model_dict:
        out["multimodal"] = {
            "framework": model_dict.get("framework", "CoCa"),
            "reconstruction_queries": model_dict.get("poolers", {}).get("reconstruction_queries", 128),
            "text_encoder_source": model_dict.get("text_encoder", {}).get(
                "source", "CONCHv1.5 pretrained text encoder"
            ),
            "text_decoder_source": model_dict.get("text_decoder", {}).get(
                "source", "CONCHv1.5 pretrained multimodal decoder"
            ),
            "text_encoder_layers": model_dict.get("text_encoder", {}).get("num_layers", 12),
            "text_decoder_layers": model_dict.get("text_decoder", {}).get("num_layers", 12),
            "text_embedding_dim": model_dict.get("text_encoder", {}).get("embed_dim", _EMBED_DIM),
            "text_hidden_dim": model_dict.get("text_encoder", {}).get("hidden_dim", _MLP_DIM),
        }

    return out if out else model_dict


def _normalize_training_short(train_dict: Dict[str, Any], resolved_stage: str) -> Dict[str, Any]:
    """Normalize train/*.yaml stage file to canonical training section."""
    training: Dict[str, Any] = {}

    profile: Dict[str, Any] = {
        "stage": resolved_stage,
        "objective": train_dict.get("objective", ""),
        "batch_size": train_dict.get("hardware", {}).get("local_batch_size_per_gpu", 1),
        "grad_accum_steps": train_dict.get("hardware", {}).get("gradient_accumulation_steps", 1),
        "effective_batch_size": train_dict.get("hardware", {}).get("effective_batch_size", 1),
        "epochs": train_dict.get("optimization", {}).get("epochs", 1),
        "optimizer": {
            "name": train_dict.get("optimization", {}).get("optimizer", {}).get("name", "adamw"),
            "learning_rate": train_dict.get("optimization", {}).get("optimizer", {}).get("learning_rate"),
            "weight_decay": train_dict.get("optimization", {}).get("optimizer", {}).get("weight_decay"),
            "betas": train_dict.get("optimization", {}).get("optimizer", {}).get("betas"),
            "scheduler": train_dict.get("optimization", {}).get("scheduler", {}).get("name"),
            "warmup": train_dict.get("optimization", {}).get("scheduler", {}).get("warmup"),
        },
    }

    if resolved_stage == "stage1_titan_v":
        profile.update(
            {
                "iterations": train_dict.get("optimization", {}).get("iterations", 91260),
                "view_sampling": {
                    "global_views": train_dict.get("views", {}).get("global_views", 2),
                    "global_view_grid_size": train_dict.get("views", {}).get("global_view_grid_size", [14, 14]),
                    "local_views": train_dict.get("views", {}).get("local_views", 10),
                    "local_view_grid_size": train_dict.get("views", {}).get("local_view_grid_size", [6, 6]),
                },
                "augmentations": [
                    key
                    for key, enabled in train_dict.get("augmentations", {}).items()
                    if isinstance(enabled, bool) and enabled
                ]
                or ["horizontal_flip", "vertical_flip", "feature_posterization"],
                "ibot": train_dict.get("ibot", {}),
            }
        )
    elif resolved_stage == "stage2_roi_caption_alignment":
        profile.update(
            {
                "num_pairs": train_dict.get("data", {}).get("num_pairs", 423122),
                "losses": {
                    "contrastive_weight": train_dict.get("losses", {}).get("contrastive_weight"),
                    "caption_weight": train_dict.get("losses", {}).get("captioning_weight"),
                },
            }
        )
    elif resolved_stage == "stage3_wsi_report_alignment":
        profile.update(
            {
                "num_pairs": train_dict.get("data", {}).get("num_pairs", 182862),
                "notes": train_dict.get("optimization", {}).get(
                    "notes",
                    [
                        "Use smaller learning rate and weight decay for vision backbone",
                        "Use slow warm-up for vision backbone",
                    ],
                ),
                "losses": {
                    "contrastive_weight": train_dict.get("losses", {}).get("contrastive_weight"),
                    "caption_weight": train_dict.get("losses", {}).get("captioning_weight"),
                },
            }
        )

    training[resolved_stage] = profile
    return training


__all__ = [
    "ConfigLoadError",
    "ConfigValidationError",
    "RuntimeConfig",
    "HardwareConfig",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
]
