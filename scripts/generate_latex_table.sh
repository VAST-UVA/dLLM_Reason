#!/usr/bin/env bash
# generate_latex_table wrapper — forwards all arguments
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/generate_latex_table.py" "$@"
