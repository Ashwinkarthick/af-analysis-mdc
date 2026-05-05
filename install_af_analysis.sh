#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_MODE="${AF_ANALYSIS_INSTALL_MODE:-editable}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m pip install -U pip
if [ "$INSTALL_MODE" = "editable" ]; then
  "$PYTHON_BIN" -m pip install -e .
else
  "$PYTHON_BIN" -m pip install .
fi

if ! command -v alphajudge >/dev/null 2>&1; then
  echo "AlphaJudge command was not found after package install; installing from GitHub..."
  "$PYTHON_BIN" -m pip install "alphajudge @ git+https://github.com/KosinskiLab/AlphaJudge.git@main"
fi

if command -v alphajudge >/dev/null 2>&1; then
  echo "AlphaJudge is available: $(command -v alphajudge)"
else
  echo "WARNING: AlphaJudge is still not on PATH. Check the active Python environment." >&2
fi

echo "AF-Analysis installation complete. Test with: af-analysis --help"
