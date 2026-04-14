#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

if command -v pyenv >/dev/null 2>&1 && pyenv prefix 3.11.11 >/dev/null 2>&1; then
  PYENV_311="$(pyenv prefix 3.11.11)/bin/python3.11"
  if [ -x "$PYENV_311" ]; then
    PYTHON_BIN="$PYENV_311"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

echo "=== ClickLess AI Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "Python: $PYTHON_BIN"

mkdir -p \
  "$PROJECT_ROOT/data/raw" \
  "$PROJECT_ROOT/data/processed" \
  "$PROJECT_ROOT/data/neo4j_data" \
  "$PROJECT_ROOT/data/neo4j_logs" \
  "$PROJECT_ROOT/data/preferences" \
  "$PROJECT_ROOT/evaluation/results"

echo "--- Creating virtual environment ---"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "--- Installing Python packages ---"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

echo "--- Installing Playwright browser ---"
python -m playwright install chromium

if [ ! -f "$PROJECT_ROOT/.env" ]; then
  cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
  echo "Created .env from .env.example"
fi

echo "=== Setup complete ==="
echo "Activate the environment with:"
echo "  source \"$VENV_DIR/bin/activate\""
echo
echo "Optional next steps:"
echo "  1. Download datasets:"
echo "     bash data/scripts/download_instacart.sh"
echo "     bash data/scripts/download_openfoodfacts.sh"
echo "  2. Preprocess data:"
echo "     python data/scripts/preprocess_instacart.py"
echo "     python data/scripts/preprocess_off.py"
echo "  3. Start local services if you want full LLM/KG features:"
echo "     ollama serve"
echo "     docker run --name clickless-neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/clickless123 neo4j:5"
echo "  4. Launch the app:"
echo "     streamlit run src/ui/app.py --server.port=8501"
