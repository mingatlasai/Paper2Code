Updated `README.md` with the requested logic analysis, aligned to the paper, plan, design interfaces, task list, and `config.yaml`.

It now includes:
- execution order (`preprocess -> pretrain -> embed -> eval`)
- environment setup and exact version constraints
- config-driven source-of-truth rules
- data assumptions/contracts (manifest, splits, patient aggregation)
- Tier A (full) vs Tier B (public-only) reproduction scope
- evaluation/statistical logic
- known limitations and unresolved ambiguities
- implementation guardrails for the coding phase

File changed: `README.md`