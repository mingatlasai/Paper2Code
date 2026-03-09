#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] scripts/run_molecular_codex.sh is kept for compatibility. Use scripts/run_pdf.sh instead."
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_pdf.sh" "$@"
