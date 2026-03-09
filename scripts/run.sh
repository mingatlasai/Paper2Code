#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] scripts/run.sh is kept for compatibility. Use scripts/run_json.sh instead."
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_json.sh" "$@"
