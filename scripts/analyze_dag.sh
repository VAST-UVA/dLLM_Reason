#!/usr/bin/env bash
# analyze_dag wrapper — forwards all arguments
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/analyze_dag.py" "$@"
