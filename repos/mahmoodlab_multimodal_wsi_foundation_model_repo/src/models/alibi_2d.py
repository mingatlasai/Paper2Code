Implemented complete `ALiBi2D` in [src/models/alibi_2d.py](/Users/mingatlas/Projects/Paper2Code/src/models/alibi_2d.py).

What’s included:
- `ALiBi2D(nn.Module)` with required interface:
  - `__init__(num_heads: int, slopes: torch.Tensor)`
  - `build_bias(coords_xy: torch.Tensor) -> torch.Tensor`
  - `apply(attn_scores: torch.Tensor, bias: torch.Tensor) -> torch.Tensor`
- Strict validation for:
  - `num_heads`, slope shape/values, coordinate shape, attention/bias shape compatibility
- 2D Euclidean ALiBi bias construction:
  - supports `[T,2] -> [H,T,T]` and `[B,T,2] -> [B,H,T,T]`
- Safe additive bias application:
  - supports `[H,T,T]` or `[B,H,T,T]` bias (with `B=1` broadcast support)
  - dtype/device aligned to `attn_scores`
- Config-aligned constants embedded (patch size `512`, magnification `20x`, heads `12`, etc.).

Validation done:
- `python -m py_compile src/models/alibi_2d.py` succeeded.
- Runtime smoke test couldn’t run because `torch` is not installed in this shell environment.