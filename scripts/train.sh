#!/usr/bin/env bash
# train wrapper — forwards all arguments
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python "$SCRIPT_DIR/train.py" "$@"
