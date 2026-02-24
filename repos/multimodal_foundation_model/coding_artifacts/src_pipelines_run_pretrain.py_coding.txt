Implemented `src/pipelines/run_pretrain.py` as a full executable pretraining pipeline (replacing the analysis stub), including:

- Stage entrypoint `run_pretrain(cfg_or_path="configs/default.yaml")`
- Config resolution/validation via `ExperimentConfig`
- Unified logging + deterministic seeding
- Full THREADS component wiring:
  - `ConchPatchEncoder`
  - `ThreadsSlideEncoder`
  - `ScGPTRNAEncoder`
  - `DNAMLPEncoder`
  - `ContrastiveLoss`
  - `ThreadsModel`
  - `PretrainModule`
  - `ThreadsPretrainDataModule`
- Trainer setup (DDP/GPU/precision resolution with strict checks)
- RankMe callback integration
- Checkpoint manager integration:
  - resume path resolution
  - save last/final
  - best-rank checkpoint persistence
- Paper/config invariant enforcement and explicit fail-fast errors for unresolved critical values (notably contrastive temperature and AMP dtype when AMP mode is requested).

Validation run:
- `python -m py_compile src/pipelines/run_pretrain.py` passed.