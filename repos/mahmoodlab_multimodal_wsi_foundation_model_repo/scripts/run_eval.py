Implemented `scripts/run_eval.py` as a complete executable evaluation orchestrator.

### What it now does
- Loads and validates config via `ExperimentConfig.from_yaml(...)` with fallback from `config.yaml` to `configs/eval/linear_probe.yaml`.
- Enforces paper/config invariants (`512`, `20x`, `768`, eval protocol constants).
- Resolves checkpoint, inspects capability (vision-only vs multimodal), and gates language tasks accordingly.
- Builds models via `Registry`, loads checkpoint weights, freezes eval models.
- Uses `EmbeddingService` for slide/report embeddings and reuse.
- Runs all required tasks:
  - `linear_probe`
  - `knn_probe`
  - `few_shot`
  - `zero_shot`
  - `retrieval`
  - `cross_modal_retrieval`
  - `report_generation`
  - `survival`
- Writes per-task JSON artifacts and consolidated report:
  - `outputs/eval/eval_report.json`
  - `outputs/eval/eval_task_summary.csv`

### Validation
- `python -m py_compile scripts/run_eval.py` passed.

### File
- [scripts/run_eval.py](/Users/mingatlas/Projects/Paper2Code/scripts/run_eval.py)