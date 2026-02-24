Implemented complete executable pytest coverage in [tests/test_losses.py](/Users/mingatlas/Projects/Paper2Code/tests/test_losses.py).

What’s included:
- `IBOTLoss` tests:
  - finite scalar output
  - shape contract failures
  - mask contribution behavior
  - EMA center update rule
  - center update on `compute`
  - temperature sensitivity
- `contrastive_info_nce_loss` tests:
  - symmetric behavior on matched embeddings
  - higher loss under mismatched pairs
  - batch-size mismatch rejection
- `caption_cross_entropy_loss` tests:
  - ignore-index correctness
  - zero-valid-token behavior
  - invalid shape rejection
- `compute_multimodal_stage_loss` tests:
  - exact weighted-sum composition
  - zero-weight behavior
  - `None` weight default policy (`1.0`)
  - required output-key contract
- CUDA AMP safety test (auto-skipped when CUDA unavailable).
- Config-locked constant sanity check (`512`, `20x`, `768`).

Validation run:
- `python -m py_compile tests/test_losses.py` passed.
- `pytest` execution could not run because `pytest` is not installed in this environment (`No module named pytest`).