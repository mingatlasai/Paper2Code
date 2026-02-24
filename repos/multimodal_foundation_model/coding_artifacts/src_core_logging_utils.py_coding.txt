Implemented `src/core/logging_utils.py` with a complete, typed logging subsystem.

### Added
- `RunPaths` dataclass for canonical run layout (`runs/{date}/{stage}/{run_id}`) with:
  - `logs`, `metrics`, `artifacts`, `checkpoints`, `splits`, `configs`, `system`
- `UnifiedLogger` class with:
  - `run_id`, `stage`, `paths`, `rank`, `world_size`, `is_primary_rank`
  - `log_event(...)`, `log_metrics(...)`, `log_artifact(...)`, `close()`
- Public module APIs:
  - `create_run_id(...)`
  - `build_run_paths(...)`
  - `ensure_run_dirs(...)`
  - `init_unified_logger(...)`
  - `init_wandb_logger(...)`
  - `init_mlflow_logger(...)`
  - `log_event(...)`, `log_metrics(...)`, `log_artifact(...)`
  - `snapshot_resolved_config(...)`
  - `snapshot_environment(...)`
  - `snapshot_paper_fidelity_contract(...)`
  - `capture_exception(...)`
  - `finalize_run(...)`
- DDP-aware behavior:
  - rank/world-size detection from env
  - rank-scoped JSONL/log filenames for non-zero ranks
  - backend tracking only on rank 0

### Validation
- Syntax check passed:
  - `python -m py_compile repos/multimodal_foundation_model_repo/src/core/logging_utils.py`

File: [logging_utils.py](/Users/mingatlas/Projects/Paper2Code/repos/multimodal_foundation_model_repo/src/core/logging_utils.py)