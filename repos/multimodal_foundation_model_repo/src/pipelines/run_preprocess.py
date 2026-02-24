Implemented `src/pipelines/run_preprocess.py` as a complete executable pipeline (replacing the prior analysis-only placeholder).

### What’s included
- Config resolution via `ExperimentConfig` with strict preprocess invariants:
  - `20x`, `patch_size=512`, `stride=512`, `overlap=0`, encoder resize `448`
- Stage bootstrap:
  - `init_unified_logger(...)`
  - deterministic `seed_everything(...)`
- End-to-end orchestration:
  - `ManifestStore.load()` + optional cohort filtering
  - `WSIReader` open/close
  - `TissueSegmenter.predict_mask(...)`
  - `PatchTiler.tile(...)` + `extract(...)`
  - `ConchPatchEncoder.encode(...)` in deterministic chunks
  - `FeatureStore.write_patch_features(...)`
- Robust per-sample error handling and stage-level failure policy
- Output artifacts:
  - sample-level report parquet: `preprocess_report.parquet`
  - summary json: `preprocess_summary.json`
- Public entrypoint:
  - `run_preprocess(cfg_or_path: Any = "configs/default.yaml") -> dict[str, Any]`

### Validation
- `python -m py_compile src/pipelines/run_preprocess.py` passed.