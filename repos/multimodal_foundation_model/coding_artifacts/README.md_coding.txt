## Code: README.md
```python
# THREADS Reproduction (Public + Full-Fidelity Roadmap)

This repository reproduces the methodology in **"Molecular-driven Foundation Model for Oncologic Pathology"** (THREADS) using the implementation contracts and configuration provided in `config.yaml`.

The repository supports two scopes:
- **Tier A (full-fidelity target):** full-paper workflow including private cohorts (requires non-public data).
- **Tier B (public-only reproduction):** same pipeline with public cohorts only (recommended default).

## 1. Reproduction Contract

This project is implementation-constrained by:
- Provided design interfaces/classes and file structure.
- Provided task list and package versions.
- Provided `config.yaml` values.

Non-negotiable defaults from `config.yaml`:
- `patch_size: 512`, `patch_stride: 512`, `target_magnification: 20x`, `overlap: 0`
- `model.slide_embedding_dim: 1024`
- `model.slide_encoder.attention_heads_main: 2`
- `model.dna_encoder.input_dim: 1673`, `output_dim: 1024`, `dropout: 0.2`
- Linear probe: logistic regression with `C=0.5`, `solver=lbfgs`, `max_iter=10000`, `class_weight=balanced`
- Survival probe: CoxNet `max_iter=10000`, alpha defaults `OS=0.07`, `PFS=0.01` with task-specific exceptions
- THREADS fine-tuning: AdamW `lr=2.5e-5`, `weight_decay=0.0`, no LLRD, no gradient accumulation, `epochs=5`, `batch_size=1`, `patches_per_batch=2048`

## 2. Environment

Paper-aligned runtime target:
- Python `3.10.12`
- PyTorch `2.3.0`
- CUDA `12.3`

Install required packages from the task specification (version-pinned):
- `torch==2.3.0`, `torchvision==0.18.0`, `pytorch-lightning==2.3.3`
- `hydra-core==1.3.2`, `omegaconf==2.3.0`
- `timm==1.0.7`, `open-clip-torch==2.24.0`
- `openslide-python==1.3.1`, `h5py==3.11.0`
- `numpy==1.26.4`, `pandas==2.2.2`, `polars==1.4.1`, `pyarrow==17.0.0`
- `scikit-learn==1.5.1`, `scikit-survival==0.23.0`, `scipy==1.13.1`
- `statsmodels==0.14.2`, `lifelines==0.29.0`
- `segmentation-models-pytorch==0.3.3`, `monai==1.3.2`, `opencv-python-headless==4.10.0.84`
- `einops==0.8.0`, `matplotlib==3.9.1`, `seaborn==0.13.2`
- `wandb==0.17.6`, `mlflow==2.14.3`
- `PyYAML==6.0.2`, `jsonschema==4.23.0`, `tqdm==4.66.5`, `rich==13.7.1`

## 3. Repository Execution Order

Run stages in this order:
1. `preprocess`: WSI -> tissue mask -> 512x512 patches @20x -> patch features
2. `pretrain`: multimodal THREADS pretraining (WSI<->RNA/DNA contrastive)
3. `embed`: slide and patient embedding export
4. `eval`: linear probing, survival probing, retrieval, prompting, statistics

Entry point:
- `main.py`

Pipeline modules:
- `src/pipelines/run_preprocess.py`
- `src/pipelines/run_pretrain.py`
- `src/pipelines/run_embed.py`
- `src/pipelines/run_eval.py`

## 4. Config Layout

Top-level config:
- `configs/default.yaml`

Sub-config groups:
- Data: `configs/data/pretrain_public.yaml`, `configs/data/downstream_public.yaml`
- Model: `configs/model/threads.yaml`
- Train: `configs/train/pretrain.yaml`, `configs/train/finetune.yaml`
- Eval: `configs/eval/linear_probe.yaml`, `configs/eval/survival.yaml`

`config.yaml` is the paper-grounded parameter reference. Runtime YAMLs must remain consistent with these values.

## 5. Data Contracts

### 5.1 Manifest schema
All dataset flow uses `ManifestRecord` and must include:
- `sample_id`, `patient_id`, `cohort`
- `slide_path`, `magnification`
- `rna_path`, `dna_path`
- `task_labels`, `meta`

### 5.2 Split schema
All split files must preserve:
- `task_name`, `fold_id`
- `train_ids`, `test_ids`
- patient/label stratification metadata

Split policy from paper/config:
- Official single-fold datasets: `EBRAINS`, `PANDA`, `IMP`
- Otherwise: 5-fold 80:20 CV or 50-fold MC/bootstrap depending on task
- Few-shot values: `k in {1, 2, 4, 8, 16, 32}`

### 5.3 Feature store schema
Feature artifacts must include:
- `sample_id`
- patch `coords`
- patch `features`
- `encoder_name`
- `precision`

### 5.4 Patient aggregation rule
Patient-level embedding uses **union of all patches across all patient WSIs** (THREADS protocol), not simple average of slide embeddings.

## 6. Model + Training Summary

### 6.1 Preprocessing
- Tissue segmentation: FPN from `segmentation-models-pytorch` (paper states in-house checkpoint unavailable)
- Tiling: non-overlapping `512x512` at `20x`
- Patch encoder input: resize `512 -> 448`, ImageNet mean/std normalization

### 6.2 THREADS architecture
- Slide encoder: gated ABMIL, main multi-head setting = `2`
- Slide embedding dimension = `1024`
- RNA encoder: scGPT branch to `1024` projection
- DNA encoder: 4-layer MLP (`1673 -> 1024`, dropout `0.2`)
- Objective: cross-modal contrastive (InfoNCE-style)

### 6.3 Pretraining defaults
- Hardware target: `4 x A100 80GB`, DDP, AMP
- Batch size per GPU: `300`
- Max epochs: `101`
- Warmup epochs: `5`
- Peak LR: `1e-5`
- Model selection: RankMe-based checkpointing (`eps=1e-7`, after warmup)

### 6.4 Embedding extraction defaults
- Use all patches (no sampling)
- Threads extraction precision target: `bf16`
- Hardware reference: `1 x 3090Ti 24GB`

### 6.5 Fine-tuning defaults
- THREADS: 5 epochs, batch size 1, 2048 patches/batch, final checkpoint (no early stop)
- GIGAPATH baseline: official recipe (frozen patch encoder, grad accumulation 32, WD 0.05, LLRD 0.95)
- CHIEF baseline: THREADS-like fine-tuning recipe (fp32)
- ABMIL supervised baseline: 20 epochs, train random 2048 patches, test with all patches

## 7. Evaluation + Statistics

Task-family metrics:
- Binary classification: macro-AUC
- Multi-class subtyping: balanced accuracy
- Grading: quadratic weighted Cohen's kappa
- Survival: c-index

Uncertainty and significance:
- Multi-fold: mean +/- SE
- Single-fold: 95% CI via bootstrap (100 replicates)
- Per-task significance: two-way ANOVA then Tukey HSD if `p < 0.05`
- Across tasks/families: mixed-effects model with dataset random effect
- Kaplan-Meier comparison: log-rank test

## 8. Reproducibility Rules

- Seed setup must run at program start (`src/utils/seeding.py`).
- Config-driven construction only (no hidden hardcoded hyperparameter overrides).
- Persist manifests, splits, checkpoints, and metrics artifacts.
- Keep precision modes explicit per model/stage.
- Keep deterministic fold/sample generation for few-shot and bootstrap routines.

## 9. Public vs Full Reproduction Scope

### Tier A: Full-paper parity
Requires private MGH/BWH and internal cohorts, plus in-house tissue segmentation checkpoint.

### Tier B: Public-only reproduction
Uses public datasets (e.g., TCGA/GTEx/CPTAC/EBRAINS/BRACS/BCNB/PANDA/IMP). This is a **methodological reproduction** with explicitly partial benchmark coverage.

## 10. Known Limitations

From provided materials:
- Some supplemental hyperparameters are partially unreadable (certain AdamW/scheduler details).
- Exact contrastive weighting details may be incomplete in OCR text.
- Exact RankMe tie-break/save rule details may require final supplement verification.
- Full 54-task parity is blocked without private data and in-house assets.

## 11. Minimal Stage Commands

Example stage dispatch:

```bash
python main.py --stage preprocess --config configs/default.yaml
python main.py --stage pretrain --config configs/default.yaml
python main.py --stage embed --config configs/default.yaml
python main.py --stage eval --config configs/default.yaml
```

## 12. File Responsibilities (Design-Linked)

Core:
- `src/core/config.py`: config loading/validation
- `src/core/registry.py`: factory registry
- `src/core/logging_utils.py`: logging/tracking helpers

Data:
- `src/data/manifest_schema.py`, `src/data/manifest_store.py`, `src/data/split_manager.py`
- `src/data/wsi_reader.py`, `src/data/tissue_segmenter.py`, `src/data/patch_tiler.py`, `src/data/feature_store.py`, `src/data/datamodules.py`

Models:
- `src/models/patch_encoder.py`, `src/models/slide_encoder_threads.py`
- `src/models/rna_encoder_scgpt.py`, `src/models/dna_encoder_mlp.py`
- `src/models/losses.py`, `src/models/threads_model.py`

Training:
- `src/train/pretrain_module.py`, `src/train/finetune_module.py`
- `src/train/callbacks_rankme.py`, `src/train/checkpointing.py`

Evaluation:
- `src/eval/embedding_export.py`, `src/eval/linear_probe.py`, `src/eval/survival_eval.py`
- `src/eval/retrieval_eval.py`, `src/eval/prompting_eval.py`, `src/eval/stats_tests.py`

Pipelines:
- `src/pipelines/run_preprocess.py`, `src/pipelines/run_pretrain.py`
- `src/pipelines/run_embed.py`, `src/pipelines/run_eval.py`

Tests:
- `tests/test_split_manager.py`
- `tests/test_slide_encoder_shapes.py`
- `tests/test_linear_probe_pipeline.py`

---

This README is the reproducibility contract for implementation and evaluation behavior in this repository.
```