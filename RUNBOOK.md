# ClickLess AI — Demo Runbook

End-to-end process to bring the system up from a clean Sol GPU node and produce the artifacts/screenshots needed for the presentation. Assumes the repo is checked out at `/scratch/smehta90/Clickless AI` and the preprocessed datasets in `data/processed/` exist.

## 0. Environment bring-up

See [STEPS.md](STEPS.md) for the exact module loads and service-start commands. After running those, you should have:

- `mamba` env `venv` activated
- `cuda-12.8.1-gcc-12.1.0` loaded
- `ollama/0.20.4` loaded and `ollama-start` running (CUDA backend, ~5GB VRAM after first model use)
- Apptainer-Neo4j running on `bolt://localhost:7687` (HTTP UI on `:7474`)
- `INSTACART_API_KEY` **unset** (forces the mock client)

Sanity check:

```bash
hostname                                            # should match $OLLAMA_HOST host
curl -s "http://$OLLAMA_HOST/api/tags" >/dev/null && echo "ollama OK"
curl -s http://localhost:7474 | head                # neo4j discovery JSON
```

---

## 1. Cached artifacts (already produced)

These live in `data/processed/` and survive across sessions. Do **not** re-run unless you want to regenerate them.

| Artifact | Source | Notes |
|---|---|---|
| `transactions.pkl` | preprocess_instacart.py | 100k Instacart 2017 baskets (subset) |
| `product_features.parquet` | preprocess_instacart.py | 49,688 products with reorder_rate, aisle, dept |
| `off_enriched.parquet` | preprocess_off.py | 4,107,390 OFF products with nutrition |
| `association_rules.pkl` | apriori_miner.mine_rules | 116 rules at min_support=0.003, min_confidence=0.2 |
| `logistic_ranker.pkl` | logistic_ranker.train | 4-feature logistic, propensity-balanced labels |
| `triples.jsonl` | spo_extractor.extract_from_off_dataset | SPO triples extracted by Mistral 7B (1000-product subset) |

---

## 2. One-time bootstrap (only if rebuilding artifacts)

Run from the project root with `venv` active and Ollama up.

### 2a. Apriori rules
```bash
python -c "
import logging, time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.ranking.apriori_miner import mine_rules
t0 = time.time()
rules = mine_rules(min_support=0.003, min_confidence=0.2)
print(f'elapsed: {time.time()-t0:.1f}s, rules: {len(rules)}')
print(rules[['antecedents','consequents','support','confidence','lift']].head(15).to_string())
"
```
Expected: ~9s, 116 rules, top-15 by lift dominated by organic produce co-purchase patterns.

### 2b. Logistic ranker
```bash
rm -f data/processed/logistic_ranker.pkl
python -c "
import logging, time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.ranking.logistic_ranker import train, rank_products
from src.api.product_schema import Product
t0 = time.time()
train()
print(f'elapsed: {time.time()-t0:.1f}s')

candidates = [
    Product(instacart_id='1', name='Organic Whole Milk', price=4.99, availability=True, platform='test', reorder_rate=0.7),
    Product(instacart_id='2', name='Almond Milk Unsweetened', price=3.49, availability=True, platform='test', reorder_rate=0.6),
    Product(instacart_id='3', name='Whole Wheat Bread', price=3.99, availability=True, platform='test', reorder_rate=0.6),
    Product(instacart_id='4', name='Organic 2% Milk', price=5.49, availability=True, platform='test', reorder_rate=0.65),
    Product(instacart_id='5', name='Organic Bananas', price=2.99, availability=True, platform='test', reorder_rate=0.8),
    Product(instacart_id='6', name='Gluten Free Bread', price=6.99, availability=True, platform='test', reorder_rate=0.55),
]
for q in ['organic milk', 'bread', 'gluten free bread', 'bananas']:
    print(f'\\nranking for \"{q}\":')
    for p, score in rank_products(q, candidates, user_budget=10.0):
        print(f'  {score:.4f}  {p.name}')
"
```
Expected: ~0.7s training, coefficients ~ `tfidf=19.5, reorder=6.3, price=-6.0, stock=3.0`. Milks rank first for milk queries, breads first for bread queries.

### 2c. SPO triples (LLM-bound, ~17 min on warm Mistral)
```bash
python -c "
import logging, time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.knowledge_graph.spo_extractor import extract_from_off_dataset, load_triples
t0 = time.time()
extract_from_off_dataset(max_products=1000, resume=True)
print(f'total elapsed: {(time.time()-t0)/60:.1f} min')
triples = load_triples()
print(f'triples.jsonl has {len(triples)} triples across {len(set(t[\"product\"] for t in triples))} products')
for t in triples[:5]:
    print(f'  ({t[\"s\"]}) --[{t[\"p\"]}]--> ({t[\"o\"]})')
"
```
Resume-safe — if it dies mid-run you can re-invoke and it'll skip already-processed products.

### 2d. Neo4j load
```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from src.knowledge_graph.neo4j_loader import load_all
load_all(max_products=5000)
"
```
Loads Product nodes from `off_enriched.parquet` plus the SPO triples (as `Entity-RELATES->Entity` edges). Pass `max_products` to cap the product load — the full 4.1M-row parquet with row-by-row iterrows+MERGE would take hours. 5000 is enough for the KG viz screenshots. Standardizes entities (2971→2774 canonical forms on 3350 triples) before batch-merging.

---

## 3. Run the Streamlit demo

```bash
unset INSTACART_API_KEY     # forces mock Instacart client
PYTHONPATH="/scratch/smehta90/Clickless AI" streamlit run src/ui/app.py --server.port=8501 --server.headless=true
```

`PYTHONPATH` is required — Streamlit exec's the script directly so the project root is not on `sys.path`, and `from src.api...` imports fail with `ModuleNotFoundError: No module named 'src'`.

Open in browser via SSH port-forward:
```bash
# from your laptop
ssh -L 8501:sg008:8501 smehta90@sol.asu.edu
# then visit http://localhost:8501
```

The chat input is wired to `run_pipeline` in [src/orchestration/graph_builder.py:113](src/orchestration/graph_builder.py#L113) → NLU (Mistral) → mock Instacart search → KG-enriched ranking → response cards. Sidebar exposes zip code, budget, dietary preferences.

### Demo queries to screenshot

- "I need gluten-free bread under $5"
- "Organic milk and eggs for the week"
- "Snacks for a party, $30 budget, no nuts"
- "Low sodium chicken broth"

---

## 4. Screenshot checklist (target 6-8)

| # | Shot | Source | Status |
|---|---|---|---|
| 1 | Streamlit chat with ranked product cards | running app, query "gluten-free bread under $5" | ✓ captured |
| 2 | Streamlit cart panel with multiple items | running app, click Add to cart on 1-3 products | ✓ captured |
| 3 | PyVis KG visualization for a product query | Streamlit KG expander (fallback subgraph — see §5e) | optional/skipped |
| 4 | Neo4j Browser graph view | `http://localhost:7474` + Cypher query below | ✓ captured |
| 5 | Apriori rules terminal output | step 2a output | ✓ captured |
| 6 | Logistic ranker coefficients + rankings | step 2b output | ✓ captured |
| 7 | LangGraph state diagram | `build_graph().get_graph().draw_mermaid()` → mermaid.live | ✓ captured |
| 8 | Code snippet crops | [src/llm/ollama_client.py](src/llm/ollama_client.py), [src/llm/credential_vault.py](src/llm/credential_vault.py), [src/ranking/kg_ranker.py](src/ranking/kg_ranker.py) | ✓ captured |
| + | Mock checkout JSON (Browser Handoff slide) | step 4a below | ✓ captured |

Cypher query for shot #4:
```cypher
MATCH (e:Entity)-[r:RELATES]-(o:Entity) RETURN e,r,o LIMIT 50
```

### 4a. Mock checkout JSON (Browser Handoff slide)
Run in a separate terminal so Streamlit can keep running:
```bash
PYTHONPATH="/scratch/smehta90/Clickless AI" python -c "
import asyncio, json
from src.api.instacart_mock import MockInstacartClient

async def main():
    client = MockInstacartClient()
    products = await client.search_products('bread', limit=2)
    cart_items = [{'product_id': p.instacart_id, 'quantity': 1} for p in products]
    cart = await client.create_cart(cart_items)
    print(json.dumps(cart, indent=2, default=str))

asyncio.run(main())
"
```
Returns `{cart_id, cart_url, item_count}` — a mock Instacart checkout URL that stands in for the Browser Handoff demo without needing the Playwright loop.

---

## 5. Patches applied during this bring-up

These are committed in the working tree; document them so they survive code-review or rebase.

### 5a. `src/ranking/logistic_ranker.py`
Rewrote `_build_training_data`. The original had three independent bugs that caused every product to score `1.0000` at inference:

- `price_ratio` was a constant `np.ones(len(balanced))` → coefficient learned to 0.
- `in_stock_prob` and `reorder_rate` were assigned the same `balanced["reorder_rate"]` array → identical coefficients, redundant feature.
- TF-IDF cosine was computed against a single fixed query string `"grocery food product"` at training time but real queries at inference, creating a distribution shift the tiny coefficient couldn't bridge.
- The label `(reorder_rate >= 0.5)` was literally one of the features, so the model collapsed onto reorder_rate and ignored everything else.

Fix: synthetic per-row queries (50% own product name for self-match, 50% shuffled other-product name for low-match), random `price_ratio` and `in_stock_prob` distributions, and a multivariate propensity-based label that combines all four features. Also added `from sklearn.preprocessing import normalize` and switched the cosine computation to vectorized row-wise multiplication.

### 5b. `src/knowledge_graph/spo_extractor.py`
`extract_triples_from_text` now accepts both shapes returned by Mistral 7B:

- `[subject, predicate, object]` (the original prompt-requested array form)
- `{"subject": ..., "predicate": ..., "object": ...}` (what Mistral actually tends to return)

The original parser only accepted the array form and silently discarded everything, producing 0 triples per product.

### 5c. `src/knowledge_graph/neo4j_loader.py`
Added `max_products` kwarg on `load_all` and `_load_product_nodes`. Without it the loader walks the full 4.1M-row OFF parquet with `df.iterrows()` + per-row MERGE under a unique-name constraint — hours wall-clock, not minutes. With `max_products=5000` the full load is sub-minute and gives enough Product nodes for the screenshots.

### 5d. `src/ui/app.py` — add-to-cart closure bug
Both code paths (history replay at line 134 and live render at line 174) passed a *factory* as the `on_add` callback:
```python
def _make_add(rp): 
    def _add(): ...
    return _add
render_results(..., on_add=_make_add)
```
And [src/ui/components/chat.py:82](src/ui/components/chat.py#L82) does `on_add(rp)`, which just invoked the factory and threw the closure away without firing it. Result: clicking "Add to cart" silently did nothing. Fixed by replacing both factories with direct `_add_handler(rp)` functions that take `rp` and execute the add logic inline.

### 5e. `src/knowledge_graph/graph_query.py` — KG viz Cypher + fallback
Two bugs:
1. `MATCH path = (p:Product {name: $name})-[*1..$depth]-(neighbor)` — Neo4j does not allow parameterized variable-length relationships. `$depth` must be a literal. Fixed by clamping `depth` to 1-5 in Python and f-string-inlining it into the Cypher; `$name` stays parameterized.
2. `Product` nodes have zero `RELATES` edges because the loader creates Products (from parquet) and Entities (from SPO triples) in disjoint namespaces. A Product-only query always returned an empty subgraph. Added two fallback queries: an Entity-substring match on product-name tokens (>3 chars), and a top-degree Entity fallback — so the viz always shows *something*. Also tightened `LIMIT 100` → `LIMIT 25-30` and capped fallback paths to 1-hop so labels remain legible at default zoom.

**Architectural caveat.** The fallback is fuzzy token matching, not a proper product-to-entity link. For a query like "Organic Oat Bread" it matches on "organic" first and never gets to "Bread", so the rendered subgraph can surface loosely-related clusters (e.g. matcha/green-tea entities that also contain "organic"). Honest for a demo screenshot, but a reviewer who asks "why is the graph about matcha?" deserves the truthful answer: Product and Entity nodes need a cross-linking pass in the loader, which is out of the 24h scope. Shot #4 (Neo4j Browser) is the better KG-credibility shot and is already captured.

---

## 6. Known issues / non-blockers

- **GraphRAG** — uses a Mistral fallback inside [src/knowledge_graph/graphrag_interface.py](src/knowledge_graph/graphrag_interface.py), not the actual Microsoft GraphRAG library. Acceptable for the demo.
- **Browser checkout** — [src/browser/checkout_agent.py](src/browser/checkout_agent.py) skeleton only. Not on the screenshot list. Use the mock cart URL screenshot instead.
- **MiniWoB++** — [src/browser/miniwob_eval.py](src/browser/miniwob_eval.py) framework only, no task execution. Not on the shot list.
- **Tests** — [tests/](tests/) are stubs. Not on the shot list.
- **Real Instacart API** — client exists at [src/api/instacart_client.py](src/api/instacart_client.py) but blocked on a 1-week developer-account approval. The mock at [src/api/instacart_mock.py:151](src/api/instacart_mock.py#L151) is a drop-in replacement that auto-activates when `INSTACART_API_KEY` is unset and serves real Instacart 2017 + Open Food Facts data through the same canonical schema.

---

## 7. Out of scope for the presentation window

- Real Instacart API integration (use the mock)
- Kroger pivot (would step on Naman's API layer)
- BrowserUse live-search against Instacart (contradicts the proposal's Section I bot-detection argument)
- Microsoft GraphRAG library swap
- Full ablation study run
- Filling in tests
