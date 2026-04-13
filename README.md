# ClickLess AI

A conversational agent that automates online grocery shopping via natural language. Users describe their needs; the system queries the Instacart API, enriches products with a Neo4j knowledge graph (Open Food Facts nutrition data), ranks results via classical baselines + LLM scoring, and returns a checkout-ready cart.

## Architecture

```
User utterance
     │
     ▼
NLU Agent (Mistral 7B)
  └─ Intent classification + slot filling
     │
     ▼
Instacart API Client / Mock
  └─ Product search + retrieval
     │
     ▼
KG-Enriched Ranker
  ├─ Apriori co-purchase suggestions
  ├─ Logistic regression baseline
  └─ GraphRAG (Neo4j + Open Food Facts)
     │
     ▼
LangGraph Orchestration (ReAct)
     │
     ▼
Streamlit UI ──► Browser Checkout (BrowserUse + Playwright)
```

## Quick Start

```bash
# 1. Activate environment (Sol HPC)
module load mamba/latest
source activate venv
export PROJ_DIR="/scratch/smehta90/Clickless AI"
cd "$PROJ_DIR"

# 2. Copy and fill in credentials
cp .env.example .env

# 3. Start Neo4j (via Apptainer, using Sol's shared image)
mkdir -p data/neo4j_data data/neo4j_logs
apptainer run --writable-tmpfs \
  --bind "$PWD/data/neo4j_data:/data" \
  --bind "$PWD/data/neo4j_logs:/logs" \
  --env NEO4J_AUTH=neo4j/clickless123 \
  /packages/apps/simg/neo4j_5.15.0-ubi8.sif &
# Bolt: localhost:7687 | HTTP: localhost:7474 | user: neo4j / pass: clickless123
# `--writable-tmpfs` is required: Neo4j's entrypoint writes to /var/lib/neo4j/conf
# at startup, and Apptainer mounts the image read-only without it.

# 4. Start Ollama (on GPU node, via Sol module)
module load ollama/0.9.6      # see `module avail ollama` for current versions
ollama-start                  # Sol wrapper; use `ollama-stop` to shut down
ollama pull mistral:7b           # ~4 GB, NLU + SPO extraction
ollama pull llama3.2-vision:11b  # ~7.9 GB, vault + simple queries + vision

# 5. Download + preprocess data (one-time)
bash data/scripts/download_instacart.sh
bash data/scripts/download_openfoodfacts.sh
python data/scripts/preprocess_instacart.py
python data/scripts/preprocess_off.py

# 6. Load knowledge graph
python -m src.knowledge_graph.neo4j_loader

# 7. Launch UI
streamlit run src/ui/app.py --server.port=8501
```

## SSH Port Forwarding (Sol → Local Browser)

```bash
ssh -L 8501:COMPUTE_NODE_HOSTNAME:8501 smehta90@sol.asu.edu
```
Then open http://localhost:8501

## Team

| Member | Responsibilities |
|--------|-----------------|
| Swayam | Browser handoff, Streamlit UI |
| Naman | Product schema, Instacart API, LangGraph orchestration |
| Chirag | Ollama client, LLM services |
| Amrit | Data preprocessing, Apriori, Logistic Ranker |
| Aviral | NLU agent |
| Shashwat | Knowledge graph, SPO extraction |

## Evaluation

Run ablation study:
```bash
python evaluation/ablation_runner.py
```

Results are written to `evaluation/results/`.
