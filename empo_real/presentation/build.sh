#!/usr/bin/env bash
# Build the Beamer deck. Regenerates figures (optional) then compiles twice
# (for slide-number / reference resolution).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Regenerate figures from the real vignette code (needs the project venv).
if [[ "${1:-}" == "--figures" ]]; then
  "$REPO_ROOT/.venv/bin/python" "$HERE/make_figures.py"
fi

cd "$HERE"
pdflatex -interaction=nonstopmode -halt-on-error empo_mbag.tex
pdflatex -interaction=nonstopmode -halt-on-error empo_mbag.tex
echo "Built: $HERE/empo_mbag.pdf"
