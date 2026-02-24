# multimodal_foundation_model Generated Repository

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
- Pipeline scripts exist under `src/pipelines/` for preprocess/pretrain/embed/eval.
- Training logic is modularized in `src/train/` (callbacks/checkpoint/module split).
- Evaluation is separated in `src/eval/` (linear probe/survival/retrieval/prompting).
- Data responsibilities are encapsulated in `src/data/` (manifest/split/feature stores).
- Config composition is organized under `configs/` with stage-specific sub-configs.

## Repository Structure
```text
.
├── README.md
├── config.yaml
├── configs
│   ├── data
│   │   ├── downstream_public.yaml
│   │   └── pretrain_public.yaml
│   ├── default.yaml
│   ├── eval
│   │   ├── linear_probe.yaml
│   │   └── survival.yaml
│   ├── model
│   │   └── threads.yaml
│   └── train
│       ├── finetune.yaml
│       └── pretrain.yaml
├── main.py
├── src
│   ├── core
│   │   ├── config.py
│   │   ├── logging_utils.py
│   │   └── registry.py
│   ├── data
│   │   ├── datamodules.py
│   │   ├── feature_store.py
│   │   ├── manifest_schema.py
│   │   ├── manifest_store.py
│   │   ├── patch_tiler.py
│   │   ├── split_manager.py
│   │   ├── tissue_segmenter.py
│   │   └── wsi_reader.py
│   ├── eval
│   │   ├── embedding_export.py
│   │   ├── linear_probe.py
│   │   ├── prompting_eval.py
│   │   ├── retrieval_eval.py
│   │   ├── stats_tests.py
│   │   └── survival_eval.py
│   ├── models
│   │   ├── dna_encoder_mlp.py
│   │   ├── losses.py
│   │   ├── patch_encoder.py
│   │   ├── rna_encoder_scgpt.py
│   │   ├── slide_encoder_threads.py
│   │   └── threads_model.py
│   ├── pipelines
│   │   ├── run_embed.py
│   │   ├── run_eval.py
│   │   ├── run_preprocess.py
│   │   └── run_pretrain.py
│   ├── train
│   │   ├── callbacks_rankme.py
│   │   ├── checkpointing.py
│   │   ├── finetune_module.py
│   │   └── pretrain_module.py
│   └── utils
│       ├── io.py
│       ├── metrics.py
│       └── seeding.py
└── tests
    ├── test_linear_probe_pipeline.py
    ├── test_slide_encoder_shapes.py
    └── test_split_manager.py
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
Inference/embedding export is supported via `embed`/`eval` pipeline paths.

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
python src/pipelines/run_preprocess.py
python src/pipelines/run_pretrain.py
python src/pipelines/run_embed.py
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
