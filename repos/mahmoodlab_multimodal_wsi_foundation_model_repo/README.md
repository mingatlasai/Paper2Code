# mahmoodlab_multimodal_wsi_foundation_model Generated Repository

## Project Overview
This repository is an auto-generated implementation scaffold for reproducing the paper workflow and experiments.
It includes pipeline entrypoints, model/training modules, and structured YAML configuration.

## Features and Scope
- Config-driven pipeline execution
- Stage-oriented workflow (`preprocess`, `pretrain`, `embed`, `eval`) when available
- Modular code layout under `src/`
- Limitations: generated code may need manual fixes for environment/data availability and edge cases

## Repository Interpretation
- Entry point appears to be `main.py`, likely stage-dispatch based.
- Training logic is modularized in `src/train/` (callbacks/checkpoint/module split).
- Evaluation is separated in `src/eval/` (linear probe/survival/retrieval/prompting).
- Data responsibilities are encapsulated in `src/data/` (manifest/split/feature stores).
- Config composition is organized under `configs/` with stage-specific sub-configs.

## Repository Structure
```text
.
├── config.yaml
├── configs
│   ├── data
│   │   └── public_repro.yaml
│   ├── default.yaml
│   ├── eval
│   │   ├── few_shot.yaml
│   │   ├── linear_probe.yaml
│   │   ├── report_generation.yaml
│   │   ├── retrieval.yaml
│   │   ├── survival.yaml
│   │   └── zero_shot.yaml
│   ├── model
│   │   ├── titan_multimodal.yaml
│   │   └── titan_v.yaml
│   └── train
│       ├── stage1_ibot.yaml
│       ├── stage2_coca.yaml
│       └── stage3_coca.yaml
├── environment.yml
├── main.py
├── scripts
│   ├── prepare_public_data.py
│   ├── run_eval.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   └── run_stage3.py
├── src
│   ├── baselines
│   │   ├── abmil.py
│   │   └── mean_pooling.py
│   ├── core
│   │   ├── config_schema.py
│   │   ├── logging_utils.py
│   │   ├── registry.py
│   │   └── utils.py
│   ├── data
│   │   ├── build_feature_grid.py
│   │   ├── caption_report_processing.py
│   │   ├── collate.py
│   │   ├── datasets.py
│   │   ├── extract_patch_features.py
│   │   ├── segment_tissue.py
│   │   ├── tile_wsi.py
│   │   ├── tissue_grouping.py
│   │   └── wsi_reader.py
│   ├── eval
│   │   ├── embed_api.py
│   │   ├── few_shot.py
│   │   ├── knn_probe.py
│   │   ├── linear_probe.py
│   │   ├── report_generation.py
│   │   ├── retrieval.py
│   │   ├── statistics.py
│   │   ├── survival.py
│   │   └── zero_shot.py
│   ├── models
│   │   ├── alibi_2d.py
│   │   ├── coca_multimodal.py
│   │   ├── ibot_heads.py
│   │   ├── losses.py
│   │   ├── text_modules.py
│   │   └── titan_encoder.py
│   └── train
│       ├── base_trainer.py
│       ├── stage1_trainer.py
│       ├── stage2_trainer.py
│       └── stage3_trainer.py
└── tests
    ├── test_alibi_2d.py
    ├── test_eval_metrics.py
    ├── test_feature_grid.py
    └── test_losses.py
```

## Prerequisites
- Python 3.9+ (3.10 recommended)
- Linux/macOS shell environment
- Optional GPU/CUDA for training stages

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
```

## Installation
```bash
# requirements.txt not found
```

## Configuration Guide
Primary config files:
- `config.yaml` (top-level run configuration)
- `configs/default.yaml` (stage defaults and composition)
- `configs/data/*`, `configs/model/*`, `configs/train/*`, `configs/eval/*` (sub-configs)

Suggested workflow:
1. Start from `config.yaml` or `configs/default.yaml`.
2. Update paths, dataset locations, output directories, and runtime stage.
3. Keep stage-specific files aligned with model/data assumptions.

## How To Run
### Data Preparation / Preprocess
```bash
python main.py --stage preprocess --config configs/default.yaml
```

### Training / Pretrain
```bash
python main.py --stage pretrain --config configs/default.yaml
```

### Embedding / Inference
```bash
python main.py --stage embed --config configs/default.yaml
```
No dedicated inference entrypoint was detected in this generated repository.

### Evaluation
```bash
python main.py --stage eval --config configs/default.yaml
```

### Additional Commands
```bash
python -m venv .venv
python main.py --stage preprocess --config configs/default.yaml
python main.py --stage pretrain --config configs/default.yaml
python main.py --stage embed --config configs/default.yaml
python main.py --stage eval --config configs/default.yaml
```

### Tests / Debug
```bash
pytest -q
```

## Reproducibility Tips
- Pin dependencies and CUDA/toolchain versions.
- Use fixed seeds in config/runtime.
- Keep input manifests and split definitions versioned.
- Save run artifacts/logs/checkpoints per stage.

## Expected Outputs
- Preprocess artifacts: feature/manifest outputs under configured data/output paths
- Training artifacts: checkpoints and logs
- Evaluation artifacts: metrics reports and summaries

## Troubleshooting
- If a stage fails, run it independently with the same config.
- Validate paths in `config.yaml` and all nested configs.
- Confirm required datasets and metadata files exist.
- Run tests (`pytest -q`) to catch regressions after edits.

## Citation
Please cite the original paper and this generated implementation workflow in your project documentation.
