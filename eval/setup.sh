#!/usr/bin/env bash
# Install the exact locked environment of this checkout (agent + findings renderers + Laminar + the experimental
# Monty backend, used only when options_json has code_backend=monty; the default repl path never imports it).
set -euo pipefail
uv sync --frozen --extra eval --extra monty --project "$EVAL_TARGET_DIR"
"$EVAL_TARGET_DIR/.venv/bin/python" --version > "$EVAL_WORKSPACE/python.version"
sha256sum "$EVAL_TARGET_DIR/uv.lock" | cut -d ' ' -f 1 > "$EVAL_WORKSPACE/dependencies.sha256"
