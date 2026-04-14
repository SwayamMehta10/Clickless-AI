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

## Local Quick Start

```bash
./setup.sh
source .venv/bin/activate

# Download and preprocess data if you want the full dataset-backed experience
bash data/scripts/download_instacart.sh
bash data/scripts/download_openfoodfacts.sh
python data/scripts/preprocess_instacart.py
python data/scripts/preprocess_off.py

# Optional: start local services for full LLM + KG support
ollama serve
docker run --name clickless-neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/clickless123 neo4j:5

# Optional: load the knowledge graph once Neo4j is running
python -m src.knowledge_graph.neo4j_loader

# Launch the app
streamlit run src/ui/app.py --server.port=8501
```

Without the downloaded datasets, the app still boots and uses fallback mock products for basic local smoke testing. Ollama and Neo4j are only needed for the full LLM-powered and knowledge-graph features.

## Sol HPC Notes

The original demo flow for Sol HPC is documented in `RUNBOOK.md` and `STEPS.md`.

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
