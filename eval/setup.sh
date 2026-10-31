#!/usr/bin/env bash
# Install the exact locked environment of this checkout (agent + findings renderers + Laminar).
set -euo pipefail
uv sync --frozen --extra eval --project "$EVAL_TARGET_DIR"
"$EVAL_TARGET_DIR/.venv/bin/python" --version > "$EVAL_WORKSPACE/python.version"
sha256sum "$EVAL_TARGET_DIR/uv.lock" | cut -d ' ' -f 1 > "$EVAL_WORKSPACE/dependencies.sha256"
