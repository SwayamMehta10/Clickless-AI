# ClickLess AI - Implementation Plan

## Context

ClickLess AI is a conversational agent that automates online grocery shopping via natural language. A user describes their needs; the system queries the Instacart API, enriches products with a Neo4j knowledge graph (Open Food Facts nutrition data), ranks results via classical baselines + LLM scoring, and returns a checkout-ready cart. Browser automation (BrowserUse + Playwright) handles only the final checkout.

Today is April 12, 2026. The proposal's Phase 1 and 2 deadlines have passed. The repo is greenfield -- only `Project Proposal.pdf` exists. The plan below is ordered by dependency and criticality so a working end-to-end demo can be built as fast as possible.

---

## 1. Project Directory Structure

```
/scratch/smehta90/Clickless AI/
├── README.md
├── requirements.txt
├── setup.sh                          # Environment bootstrap
├── .env.example                      # API keys, Neo4j creds template
├── .gitignore
├── config/
│   ├── settings.yaml                 # Endpoints, thresholds, model names
│   └── ollama_models.yaml            # Model pull list
├── data/
│   ├── raw/instacart_2017/           # Kaggle CSVs
│   ├── raw/openfoodfacts/            # OFF CSV dump
│   ├── processed/                    # association_rules.pkl, product_features.parquet, triples.jsonl, off_enriched.parquet
│   └── scripts/                      # download_instacart.sh, download_openfoodfacts.sh, preprocess_instacart.py, preprocess_off.py
├── src/
│   ├── nlu/                          # intent_classifier.py, slot_filler.py, dialogue_state.py
│   ├── api/                          # instacart_client.py, instacart_mock.py, product_schema.py
│   ├── knowledge_graph/              # spo_extractor.py, entity_standardizer.py, neo4j_loader.py, graph_query.py, graphrag_interface.py
│   ├── ranking/                      # apriori_miner.py, logistic_ranker.py, kg_ranker.py
│   ├── llm/                          # ollama_client.py, credential_vault.py, session_manager.py, preference_model.py
│   ├── browser/                      # checkout_agent.py, miniwob_eval.py
│   ├── orchestration/                # graph_builder.py, agents.py, state.py
│   └── ui/                           # app.py, components/chat.py, components/cart.py, components/kg_viz.py
├── tests/                            # test_nlu.py, test_api.py, test_kg.py, test_ranking.py, test_orchestration.py, test_browser.py
├── evaluation/                       # ablation_runner.py, metrics.py, scenarios.py, user_study_forms.py, results/
└── notebooks/                        # EDA and exploration notebooks
```

---

## 2. Environment Setup (ASU Sol Supercomputer)

All commands below are meant for **you to run manually** on Sol. Nothing is auto-executed.

### 2a. Python Environment via Conda

Sol provides `mamba` via the module system. Run these on a login node:

```bash
# Load mamba module (check exact name with: module avail mamba)
module load mamba/latest

# Create a conda environment in your scratch space
mamba create -n venv

# Activate it
source activate venv

# Install all dependencies
pip install \
  langchain langgraph langchain-community \
  streamlit neo4j pyvis ollama mlxtend \
  scikit-learn pandas pyarrow \
  browser-use playwright \
  graphrag python-dotenv pydantic httpx \
  cryptography pytest tiktoken numpy

# Install Playwright browser binaries (for BrowserUse)
playwright install chromium
```

Add to your `~/.bashrc` or create an `activate.sh` script for convenience:
```bash
#!/bin/bash
module load mamba/latest
source activate venv
export PROJ_DIR="/scratch/smehta90/Clickless AI"
cd "$PROJ_DIR"
```

### 2b. Ollama Setup on Sol

Ollama is provided as a Sol module — **do not install it manually**. It will not run on a login node; you must be on a GPU compute node first. Reference: https://docs.rc.asu.edu/ollama

```bash
# Request an interactive GPU node (adjust partition/QOS to your allocation)
salloc --partition=public --qos=public --gres=gpu:a100:1 --mem=64G --time=4:00:00

# Once on the compute node, load the Ollama module
module avail ollama          # see what versions Sol currently provides
module load ollama/0.9.6     # 0.4+ is required for llama3.2-vision

# Start the server using Sol's wrapper (NOT `ollama serve` directly)
ollama-start                 # shut down later with `ollama-stop`

# Pull models (one-time; cached under your home/scratch per Sol's defaults)
ollama pull mistral:7b              # ~4GB, NLU + SPO extraction
ollama pull llama3.2-vision:11b     # ~7.9GB, credential vault + simple queries + vision
# Optional stretch goal — only pull if you actually need it:
# ollama pull llama3.2:90b-vision-instruct-q4_K_M  # ~50GB
```

**Important Sol notes:**
- Ollama server must be running on the **same compute node** where your code runs. Start it at the beginning of each GPU session.
- Use `module load ollama/<version>` rather than the upstream install script — Sol manages the binary and GPU drivers for you.
- If the 90B model doesn't fit or takes too long to pull, deprioritize it (it's a stretch goal). The 11B + Mistral 7B are sufficient for the core demo.

For **batch jobs**, create a Slurm script that starts Ollama before your app:
```bash
#!/bin/bash
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

module load mamba
mamba activate /scratch/smehta90/clickless_env
module load ollama/0.9.6

ollama-start  # Sol wrapper; use ollama-stop in any cleanup logic

cd "/scratch/smehta90/Clickless AI"
streamlit run src/ui/app.py --server.port=8501
```

### 2c. Neo4j via Apptainer

Sol provides a prebuilt Neo4j image at `/packages/apps/simg/neo4j_5.15.0-ubi8.sif` — use it directly instead of pulling your own. Run `showsimg` on any Sol node to browse the full library. Reference: https://docs.rc.asu.edu/apptainer

```bash
# Create data directories
mkdir -p "/scratch/smehta90/Clickless AI/data/neo4j_data"
mkdir -p "/scratch/smehta90/Clickless AI/data/neo4j_logs"

# Run Neo4j on the compute node (same node as Ollama and your app)
apptainer run --writable-tmpfs \
  --bind "/scratch/smehta90/Clickless AI/data/neo4j_data:/data" \
  --bind "/scratch/smehta90/Clickless AI/data/neo4j_logs:/logs" \
  --env NEO4J_AUTH=neo4j/clickless123 \
  /packages/apps/simg/neo4j_5.15.0-ubi8.sif &
```

`--writable-tmpfs` is required: Neo4j's entrypoint writes to `/var/lib/neo4j/conf/neo4j.conf` at startup, and Apptainer mounts the SIF read-only without it. The tmpfs overlay is in-memory only — your `/data` and `/logs` binds give you persistence for the actual database and logs.

Neo4j will be available at `bolt://localhost:7687`. Set credentials in your `.env` file:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=clickless123
```

**Note:** Both Ollama and Neo4j must run on the same node as your application. For the demo, use an interactive `salloc` session so all services share one node.

### 2d. Streamlit Port Forwarding

Since Sol compute nodes aren't directly accessible from your browser, use SSH tunneling:

```bash
# From your LOCAL machine (laptop), after noting the compute node hostname:
ssh -L 8501:COMPUTE_NODE:8501 smehta90@sol.asu.edu
```

Then open `http://localhost:8501` in your browser to access the Streamlit UI.

### 2e. Key Dependencies Summary
`langchain`, `langgraph`, `langchain-community`, `streamlit`, `neo4j`, `pyvis`, `ollama`, `mlxtend` (Apriori), `scikit-learn`, `pandas`, `pyarrow`, `browser-use`, `playwright`, `graphrag`, `python-dotenv`, `pydantic`, `httpx`, `cryptography`, `pytest`, `tiktoken`

### 2f. Ollama Models Summary
| Model | Size | Purpose |
|-------|------|---------|
| `mistral:7b` | ~4GB | NLU intent/slot extraction + SPO triple extraction |
| `llama3.2:11b` | ~6GB | Credential vault, simple queries, session tokens |
| `llama3.2:90b-vision-instruct-q4_K_M` | ~50GB | Complex queries + vision (stretch goal) |

---

## 3. Implementation Steps (Dependency Order)

### Layer 0: Foundation (Day 1-2) -- All in parallel

| Task | Owner | File(s) | Details |
|------|-------|---------|---------|
| Config + Product Schema | Naman | `config/settings.yaml`, `src/api/product_schema.py` | Pydantic `Product` dataclass with: name, brand, price, availability, platform, nutriscore, nova_group, allergens, category, image_url, instacart_id |
| Ollama Client | Chirag | `src/llm/ollama_client.py` | Wrapper around `ollama` package: `generate()`, `chat()`, `embed()`. Retries, timeout, model routing. Single access point for all LLM calls. |
| Data Download + Preprocessing | Amrit | `data/scripts/*` | Download Instacart 2017 (Kaggle) and Open Food Facts CSV. Preprocess into `product_features.parquet`, `transactions.pkl`, `off_enriched.parquet`. |
| Git init + README | Swayam | `README.md`, `.gitignore` | Initialize repo, document setup steps. |

### Layer 1: Classical Baselines (Day 2-4) -- Amrit

**Apriori Mining** -- `src/ranking/apriori_miner.py`
- Load transactions from `transactions.pkl`
- `mlxtend.frequent_patterns.apriori` with `min_support=0.01`
- Generate rules with `min_confidence=0.3`
- Expose: `get_copurchase_suggestions(product_name, top_k=5) -> List[str]`

**Logistic Regression Ranker** -- `src/ranking/logistic_ranker.py`
- 4-feature vector: price_ratio, in_stock_prob, tfidf_cosine, reorder_rate
- Train on Instacart 2017 (reordered = positive, same-aisle not purchased = negative)
- Expose: `rank_products(query, candidates, user_budget) -> List[Tuple[Product, float]]`

### Layer 2: NLU Agent (Day 2-4) -- Aviral

**Intent Classifier** -- `src/nlu/intent_classifier.py`
- Mistral 7B via Ollama, structured prompt
- Intents: `search_product`, `add_to_cart`, `remove_from_cart`, `set_constraint`, `checkout`, `get_recommendation`, `chit_chat`

**Slot Filler** -- `src/nlu/slot_filler.py`
- Mistral 7B few-shot extraction: item, quantity, unit, max_price, dietary_flags, brand_preference

**Dialogue State** -- `src/nlu/dialogue_state.py`
- Pydantic `DialogueState`: current intent, filled slots, conversation history (last 10 turns), current cart, active constraints

### Layer 3: Instacart API (Day 2-5) -- Naman

**API Client** -- `src/api/instacart_client.py`
- `httpx.AsyncClient` wrapper: `search_products()`, `get_product_details()`, `get_retailers()`, `create_cart()`
- Rate limiting with exponential backoff
- Maps API responses to canonical `Product`

**Mock API** -- `src/api/instacart_mock.py`
- Fallback if real API access unavailable. Serves products from Instacart 2017 + Open Food Facts data. Critical for demos.

### Layer 4: Knowledge Graph (Day 3-7) -- Shashwat + Chirag

**SPO Extraction** -- `src/knowledge_graph/spo_extractor.py`
- Chunk Open Food Facts descriptions (200-300 words, 10% overlap)
- Mistral 7B extracts triples (predicates 1-3 words)
- Save to `triples.jsonl`

**Entity Standardizer** -- `src/knowledge_graph/entity_standardizer.py`
- TF-IDF cosine clustering (threshold 0.85) + Mistral 7B canonical name selection

**Neo4j Loader** -- `src/knowledge_graph/neo4j_loader.py`
- Batch-insert triples, add Open Food Facts attributes as node properties

**Graph Query** -- `src/knowledge_graph/graph_query.py`
- Cypher helpers: `find_related_products()`, `find_by_attribute()`, `get_product_subgraph()`

**GraphRAG** -- `src/knowledge_graph/graphrag_interface.py`
- Microsoft GraphRAG integration for NL querying with cited sources

### Layer 5: KG-Enriched Ranking (Day 6-8) -- Amrit + Shashwat

`src/ranking/kg_ranker.py`
- Combines: logistic score + KG nutritional match + Apriori co-purchase + GraphRAG relevance
- Weighted linear combination (weights in config)
- Expose: `rank_with_kg(query, candidates, constraints, user_prefs) -> List[RankedProduct]`

### Layer 6: LLM Services (Day 3-6) -- Chirag

**Credential Vault** -- `src/llm/credential_vault.py`
- `cryptography.fernet` encryption, PBKDF2 key derivation
- Llama 3.2 11B generates session tokens locally

**Preference Model** -- `src/llm/preference_model.py`
- JSON per user: preferred brands, allergens, budget, purchase history
- Updated after cart confirmation: `update_preferences(user_id, confirmed_cart, rejected_items)`

### Layer 7: LangGraph Orchestration (Day 7-10) -- Naman

**State** -- `src/orchestration/state.py`
```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    dialogue_state: DialogueState
    search_results: List[Product]
    ranked_results: List[RankedProduct]
    cart: List[CartItem]
    checkout_ready: bool
    error: Optional[str]
```

**Graph Builder** -- `src/orchestration/graph_builder.py`
- LangGraph `StateGraph` nodes: nlu_agent -> api_agent -> kg_ranking_agent -> response_generator
- Conditional routing: search intent goes full pipeline, checkout routes to browser handoff, chit_chat goes direct to response
- ReAct-style prompting (Thought -> Action -> Observation loop)

**Agent Implementations** -- `src/orchestration/agents.py`
- Each agent is a function `(AgentState) -> AgentState`

### Layer 8: Browser Handoff (Day 8-11) -- Swayam

**Checkout Agent** -- `src/browser/checkout_agent.py`
- BrowserUse + Playwright: inject session token -> navigate to cart -> verify contents -> proceed to checkout (stop before payment for demos)
- Llama 3.2 90B vision for screenshot interpretation fallback

**MiniWoB++ Eval** -- `src/browser/miniwob_eval.py`
- Validate browser agent on form-fill/navigation tasks before live checkout

### Layer 9: Streamlit UI (Day 5-10) -- Swayam

**Main App** -- `src/ui/app.py`
- Sidebar: zip code, budget, dietary preferences
- Main: chat via `st.chat_message` / `st.chat_input`
- Invokes LangGraph pipeline, streams results

**Components**:
- `chat.py`: Product cards with image, price, availability, nutriscore badge
- `cart.py`: Cart contents, running total, "Confirm and Checkout" button
- `kg_viz.py`: PyVis knowledge graph embedded via `st.components.v1.html`

---

## 4. Pragmatic Prioritization

### Must-have for demo (Week 1-2)
1. Environment setup (everyone, day 1)
2. Mock Instacart API if real API unavailable (Naman, day 1-2)
3. NLU agent with Mistral 7B (Aviral, day 1-3)
4. Apriori + logistic ranker on Instacart 2017 (Amrit, day 1-4)
5. Basic LangGraph orchestration: NLU -> API -> Ranking -> Response (Naman, day 3-6)
6. Streamlit chat UI showing ranked results (Swayam, day 3-6)
7. Basic KG with Open Food Facts attributes in Neo4j (Shashwat, day 2-5)
8. Ollama client + Llama 3.2 integration (Chirag, day 1-3)

### Should-have (Week 2-3)
9. Full SPO extraction + GraphRAG integration
10. KG-enriched ranking
11. Preference learning module
12. PyVis visualization panel
13. Ablation study (Configs A/B/C)

### Stretch goals
14. BrowserUse checkout handoff (demo with recorded video if unstable)
15. MiniWoB++ evaluation
16. Credential vault with encrypted storage
17. Llama 3.2 90B vision integration

---

## 5. Evaluation & Ablation

### Ablation Configurations
| Config | Components | Purpose |
|--------|-----------|---------|
| A | Apriori + logistic regression only | Lower-bound, no LLM/KG |
| B | ReAct + Instacart API, no KG | Isolates KG-RAG contribution |
| C | Full system (ReAct + KG-RAG + GraphRAG) | Expected best performance |

### Test Scenarios (15-20 total)
- **Weekly grocery list** (5): e.g. "I need milk, eggs, bread, chicken, broccoli under $30"
- **Dietary-restricted meal prep** (5): e.g. "Gluten-free pasta, low-sodium sauce, organic turkey"
- **Budget-capped bulk** (5): e.g. "Snacks for 20 people, $50 budget, no nuts"

### Metrics & Targets
| Metric | Target | Implementation |
|--------|--------|---------------|
| Task Success Rate (TSR) | >= 85% | All hard constraints met + valid result |
| Constraint Satisfaction Score (CSS) | >= 90% | Fraction of constraints in top result |
| NDCG@5 | -- | Ranking quality vs human annotation |
| Clicks Saved | >= 10/session | Manual vs agent for 10-item list |
| Time-to-First-Option | < 5 sec | Wall-clock query-to-result latency |
| Triple Precision | -- | Manual verification of 100 triples |
| Browser ESR | >= 90% | MiniWoB++ success ratio |

### User Study
5-7 participants, 3 scenarios each. Collect: task completion, clicks, SUS score, free-form feedback.

---

## 6. Verification Checklist

1. **Environment**: `python --version` -> 3.11.x; `ollama list` shows all 3 models; Neo4j responds to basic Cypher
2. **Data**: `product_features.parquet` has >49K products; `association_rules.pkl` has >100 rules; `off_enriched.parquet` has >100K products
3. **NLU**: "I need 2 gallons of organic whole milk under $6" -> intent=`search_product`, slots=`{item: "whole milk", quantity: 2, unit: "gallon", dietary_flags: ["organic"], max_price: 6.0}`
4. **API**: `search_products("organic milk")` returns >=5 valid `Product` objects
5. **Baselines**: Apriori suggests fruits for "banana"; logistic ranker sorts by relevance
6. **KG**: Neo4j has >10K nodes, >20K relationships; dietary queries return results
7. **Orchestration**: "Find me gluten-free bread under $5" through LangGraph returns ranked results in <10s
8. **UI**: `streamlit run src/ui/app.py` shows chat, product cards, cart panel
9. **Ablation**: Runner produces CSV with TSR/CSS/NDCG@5 across all 3 configs; Config C > B > A
10. **Browser** (if implemented): MiniWoB++ form-fill >= 90% success

---

## 7. Critical Files

These are the highest-leverage files -- errors here block everything downstream:

- `src/api/product_schema.py` -- Canonical `Product` dataclass. Every component imports this.
- `src/llm/ollama_client.py` -- Single LLM access layer used by NLU, SPO extraction, GraphRAG, preferences.
- `src/orchestration/graph_builder.py` -- LangGraph agent graph is the system's central nervous system.
- `src/api/instacart_client.py` (+ `instacart_mock.py`) -- Data source for the entire pipeline.
- `src/ui/app.py` -- Streamlit entry point; this is what evaluators see.
