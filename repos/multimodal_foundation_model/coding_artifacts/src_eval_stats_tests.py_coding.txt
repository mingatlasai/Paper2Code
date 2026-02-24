Implemented `src/eval/stats_tests.py` with a complete, design-locked `StatsAnalyzer`:

- Public interface implemented exactly:
  - `anova_two_way(results_df)`
  - `tukey_hsd(results_df)`
  - `mixed_effects(results_df)`
  - `bootstrap_ci(values, n_boot)`
- Paper/config-aligned defaults and behavior:
  - `alpha=0.05`
  - bootstrap default `100` replicates
  - ANOVA gate before Tukey
  - mixed-effects with dataset random effect and model fixed effect
- Added strict validation, deterministic bootstrap seeding, robust error handling, and typed private helpers.
- Kept log-rank as a private helper (`_logrank_test`) to avoid expanding public API.

File updated:
- `src/eval/stats_tests.py`

Validation run:
- `python -m py_compile src/eval/stats_tests.py` passed.
- Runtime smoke test could not run in this environment because `pandas` is not installed.