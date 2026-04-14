#!/bin/bash

set -e

PROJ_DIR="/scratch/smehta90/Clickless AI"
SCRATCH="/scratch/smehta90"

echo "=== ClickLess AI Setup ==="

# 1. Python env
echo "--- Python environment ---"
module load mamba/latest
source activate venv

# 2. Install dependencies
echo "--- Installing Python packages ---"
pip install -r "$PROJ_DIR/requirements.txt"

# 3. Playwright browsers
echo "--- Installing Playwright browsers ---"
playwright install chromium

# 4. Ollama (Sol module — must be on a GPU compute node, not a login node)
echo "--- Setting up Ollama ---"
module load ollama/0.9.6   # `module avail ollama` for current versions
ollama-start

# Pull models
echo "--- Pulling Ollama models ---"
ollama pull mistral:7b
ollama pull llama3.2-vision:11b
echo "Skipping 90B model -- pull manually if needed: ollama pull llama3.2:90b-vision-instruct-q4_K_M"

# 5. Neo4j via Apptainer (use Sol's prebuilt image at /packages/apps/simg/)
echo "--- Setting up Neo4j ---"
NEO4J_SIF="/packages/apps/simg/neo4j_5.15.0-ubi8.sif"

mkdir -p "$PROJ_DIR/data/neo4j_data" "$PROJ_DIR/data/neo4j_logs"

echo "To start Neo4j, copy-paste this single-line command:"
echo "  apptainer run --writable-tmpfs --bind '$PROJ_DIR/data/neo4j_data:/data' --bind '$PROJ_DIR/data/neo4j_logs:/logs' --env NEO4J_AUTH=neo4j/clickless123 $NEO4J_SIF"
echo "  # --writable-tmpfs is required so Neo4j can write to /var/lib/neo4j/conf at startup"

# 6. .env file
if [ ! -f "$PROJ_DIR/.env" ]; then
    cp "$PROJ_DIR/.env.example" "$PROJ_DIR/.env"
    echo "Created .env from .env.example -- fill in your API keys!"
fi

echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Edit .env with your Instacart API key"
echo "  2. Download data: bash data/scripts/download_instacart.sh"
echo "  3. Preprocess: python data/scripts/preprocess_instacart.py && python data/scripts/preprocess_off.py"
echo "  4. Load KG: python -m src.knowledge_graph.neo4j_loader"
echo "  5. Launch: streamlit run src/ui/app.py --server.port=8501"
