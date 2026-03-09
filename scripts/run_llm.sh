#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] scripts/run_llm.sh is kept for compatibility. Use scripts/run_json_llm.sh instead."
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_json_llm.sh" "$@"
