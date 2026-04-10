#!/usr/bin/env bash
# benchmark_schedulers wrapper — forwards all arguments
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/benchmark_schedulers.py" "$@"
