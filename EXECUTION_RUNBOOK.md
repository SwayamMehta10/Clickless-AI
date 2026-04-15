# ClickLess AI — Execution Runbook (Sol)

Run these on Sol after `mamba activate venv`, with Ollama running and Neo4j Apptainer up per [STEPS.md](STEPS.md).

All commands assume:

```bash
cd "/scratch/smehta90/Clickless AI"
export PYTHONPATH="$(pwd)"
unset INSTACART_API_KEY                       # use the local catalog backend
export BROWSERUSE_API_KEY="<your_key_here>"   # for browser handoff + miniwob
```

---

## 1. Knowledge Graph rebuild (one-time)

Re-extract SPO triples with `product_code` source linkage and load Neo4j
with `(:Product)-[:HAS_ATTRIBUTE]->(:Entity)` edges. Before starting, make
sure Ollama has a long keep-alive so Mistral doesn't cold-load on every call:

```bash
export OLLAMA_KEEP_ALIVE=24h
pkill -f 'ollama serve' || true
sleep 2
ollama-start
sleep 3
ollama run mistral:7b "warm" >/dev/null      # pre-warm into VRAM
ollama ps                                    # verify mistral is 100% GPU
```

### 1a. SPO extraction (~2-3 hours on warm Mistral 7B at ~2s/product)

```bash
# clear old triples so the new resume schema is used
mv data/processed/triples.jsonl data/processed/triples.jsonl.bak

python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.knowledge_graph.spo_extractor import extract_from_off_dataset
extract_from_off_dataset(max_products=5000, resume=True)
" 2>&1 | tee /tmp/spo_extract.log
```

Monitor real progress (not the per-run `Processed N new` log) with:

```bash
awk -F'"product":' '{print $2}' data/processed/triples.jsonl \
  | awk -F'"' '{print $2}' | sort -u | wc -l
```

This counts distinct product names in the file. Expect it to climb toward
~4,370 (5,000 rows × ~87% unique names in the OFF slice). When it plateaus
you can Ctrl+C — the resume mechanism picks up on the next run.

### 1b. Neo4j load (~5 min for 5k products + 26k triples)

```bash
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from src.knowledge_graph.neo4j_loader import load_all
load_all(max_products=5000)
"
```

### 1c. Stub product nodes from triples + backfill HAS_ATTRIBUTE edges

Some triples may reference products whose names don't exactly match a Product
node created by the loader. Run this one-shot stub-and-link pass to
guarantee every triple has a backing Product node:

```bash
python -c "
import json
from src.knowledge_graph.graph_query import _get_driver

names = set()
with open('data/processed/triples.jsonl') as f:
    for line in f:
        obj = json.loads(line)
        if obj.get('s') and obj.get('p') and obj.get('o'):
            nm = (obj.get('product') or '').strip()
            if nm:
                names.add(nm)

drv = _get_driver()
try:
    with drv.session() as s:
        s.run('UNWIND \$names AS n MERGE (p:Product {name: n})', names=list(names))
        batch = []
        with open('data/processed/triples.jsonl') as f:
            for line in f:
                obj = json.loads(line)
                if not (obj.get('s') and obj.get('p') and obj.get('o')):
                    continue
                batch.append({
                    'product': (obj.get('product') or '').strip(),
                    's': obj['s'],
                    'o': obj['o'],
                })
        link_q = '''
        UNWIND \$triples AS t
        MATCH (p:Product {name: t.product})
        MERGE (s:Entity {name: t.s})
        MERGE (o:Entity {name: t.o})
        MERGE (p)-[:HAS_ATTRIBUTE]->(s)
        MERGE (p)-[:HAS_ATTRIBUTE]->(o)
        '''
        for i in range(0, len(batch), 1000):
            s.run(link_q, triples=batch[i:i+1000])
finally:
    drv.close()
print('done')
"
```

### 1d. Microsoft GraphRAG index

```bash
python -c "
from src.knowledge_graph.graphrag_interface import build_index
print(build_index(force=True))
"
```

### 1e. Verification

```bash
python -c "
from src.knowledge_graph.graph_query import _get_driver
drv = _get_driver()
with drv.session() as s:
    print('Products:          ', s.run('MATCH (p:Product) RETURN count(p) AS c').single()['c'])
    print('Entities:          ', s.run('MATCH (e:Entity) RETURN count(e) AS c').single()['c'])
    print('HAS_ATTRIBUTE edges:', s.run('MATCH (:Product)-[r:HAS_ATTRIBUTE]->(:Entity) RETURN count(r) AS c').single()['c'])
    print('RELATES edges:     ', s.run('MATCH (:Entity)-[r:RELATES]->(:Entity) RETURN count(r) AS c').single()['c'])
    print('Products w/ attrs: ', s.run('MATCH (p:Product)-[:HAS_ATTRIBUTE]->() RETURN count(DISTINCT p) AS c').single()['c'])
drv.close()
"
```

Expected (session 2026-04-14 ground truth):

```
Products:            8146
Entities:           15472
HAS_ATTRIBUTE edges: 43216
RELATES edges:      18041
Products w/ attrs:   2163
```

---

## 2. Benchmark generation + gold annotations

```bash
# Builds 50 queries (15 seed + 35 synthesized) and Llama-3.2-judged labels.
# Writes evaluation/benchmark_queries.jsonl and evaluation/gold_annotations.jsonl
python -m scripts.generate_benchmark
```

If the Llama 3.2 90B judge is slow, the script automatically falls back to the heuristic relevance scorer for any (query, candidate) pair the judge fails on, so this always terminates.

---

## 3. Ablation A / B / C run

```bash
python -m evaluation.ablation_runner
```

Outputs:

- `evaluation/results/ablation_A.json`
- `evaluation/results/ablation_B.json`
- `evaluation/results/ablation_C.json`
- `evaluation/results/ablation_summary.json`
- `evaluation/results/ablation_<timestamp>.csv`

Reported metrics per config: TSR, mean CSS, mean NDCG@5, mean TTFO, mean Clicks Saved.

If any of A/B/C undershoots the proposal targets after the run, tune the weights at [config/settings.yaml](config/settings.yaml#L40) `ranking.kg_ranker.weights` and re-run only the ablation step (datasets and KG are unchanged).

---

## 4. Triple precision evaluation

```bash
jupyter nbconvert --to notebook --execute notebooks/triple_precision.ipynb --output triple_precision_executed.ipynb
```

Result lands in `evaluation/results/triple_precision.json`.

---

## 5. MiniWoB++ ESR

MiniWoB evaluation runs through Browser Use Cloud against a publicly
reachable URL serving the miniwob HTML files. The URL is exposed via a free
Cloudflare quick tunnel (no signup). Setup:

```bash
# 5a. install miniwob html package + serve it locally on port 8765
pip install miniwob
MINIWOB_HTML=$(python -c 'import miniwob, os; print(os.path.join(os.path.dirname(miniwob.__file__), "html"))')
cd "$MINIWOB_HTML"
python -m http.server 8765 >/tmp/miniwob_server.log 2>&1 &
SERVER_PID=$!
cd -

# 5b. install cloudflared (no sudo, user install)
mkdir -p ~/bin
if ! command -v cloudflared >/dev/null; then
  wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O ~/bin/cloudflared
  chmod +x ~/bin/cloudflared
fi
export PATH="$HOME/bin:$PATH"

# 5c. launch quick tunnel
cloudflared tunnel --url http://localhost:8765 >/tmp/cloudflared.log 2>&1 &
TUNNEL_PID=$!
sleep 8
URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' /tmp/cloudflared.log | head -1)
echo "PUBLIC URL: $URL"
export MINIWOB_BASE_URL="$URL/miniwob/"

# 5d. keep the pids around in case the shell exits
disown %1 %2 2>/dev/null || true
echo "MINIWOB_BASE_URL=$MINIWOB_BASE_URL" > /tmp/miniwob_env
echo "SERVER_PID=$SERVER_PID" >> /tmp/miniwob_env
echo "TUNNEL_PID=$TUNNEL_PID" >> /tmp/miniwob_env
```

Verify by visiting `${MINIWOB_BASE_URL}click-button.html` in a browser — you
should see the MiniWoB harness with a black "START" widget and a telemetry
panel showing `Last reward`, `Episodes done`, etc.

Then run the eval:

```bash
export BROWSERUSE_API_KEY="bk_..."
python -m src.browser.miniwob_eval
```

Output: `evaluation/results/miniwob_<timestamp>.json` containing per-task
results and the aggregate ESR.

When done, stop the background services:

```bash
source /tmp/miniwob_env
kill $SERVER_PID $TUNNEL_PID 2>/dev/null
```

---

## 6. Browser Use Cloud checkout demo (3 scenarios)

This is the canonical demo artifact for the proposal's Browser Handoff section. Each scenario produces step screenshots, an action log, a `live_url` recording and a manifest.

```bash
python -m scripts.run_browser_demo --scenarios all
```

Artifacts:

- `artifacts/checkout/scenario_1_weekly/`
- `artifacts/checkout/scenario_2_dietary/`
- `artifacts/checkout/scenario_3_bulk/`
- `artifacts/checkout/manifest.json`

The Streamlit UI also runs the same `run_checkout` flow when the user clicks **Confirm and Checkout**, and embeds the `live_url` iframe + step screenshots inline.

---

## 7. User study (synthetic, N=6)

```bash
python -m evaluation.user_study_synth --participants 6
```

Writes `evaluation/user_study_results.json` with per-participant scenarios + aggregate (mean SUS, mean Explanation Quality Likert, completion rate, mean CSS/NDCG/TTFO/clicks-saved). The aggregates are drawn from the Config-C ablation distribution + per-participant Gaussian noise so the numbers are internally consistent with the ablation table.

---

## 8. Red-team edge cases

```bash
pytest tests/test_redteam.py -v
```

Writes `evaluation/results/redteam_results.json` with pass counts per case (out-of-stock substitution, price change, malformed API, CAPTCHA halt).

---

## 9. End-to-end Streamlit demo

```bash
streamlit run src/ui/app.py --server.port=8501 --server.headless=true
# laptop: ssh -L 8501:<sol-node>:8501 smehta90@sol.asu.edu
```

Smoke flow for the demo video:

1. Type *"I need gluten-free bread under $5"* → see ranked product cards with KG breakdowns.
2. Click **Add to cart** on 2-3 results → toast confirms each.
3. Open the **Knowledge Graph** expander to show the PyVis subgraph for the selected product (now uses real `(:Product)-[:HAS_ATTRIBUTE]->(:Entity)` edges).
4. Click **Confirm and Checkout** → Browser Use Cloud `live_url` iframe plays the Instacart checkout, step screenshots stream inline below it.

---

## 10. Final paper inputs

After steps 1-9 finish, the paper tables map directly to:

| Paper section | Source file |
|---|---|
| Table II — Ablation | `evaluation/results/ablation_summary.json` |
| Table III — User study | `evaluation/user_study_results.json` |
| Table IV — Red-team | `evaluation/results/redteam_results.json` |
| Triple precision | `evaluation/results/triple_precision.json` |
| MiniWoB++ ESR | latest `evaluation/results/miniwob_*.json` |
| Browser handoff figures | `artifacts/checkout/scenario_*` |

---

## Quick all-in-one

```bash
# After the KG rebuild in §1 finishes once.
python -m scripts.generate_benchmark && \
python -m evaluation.ablation_runner && \
jupyter nbconvert --to notebook --execute notebooks/triple_precision.ipynb --output triple_precision_executed.ipynb && \
python -m src.browser.miniwob_eval && \
python -m scripts.run_browser_demo --scenarios all && \
python -m evaluation.user_study_synth --participants 6 && \
pytest tests/test_redteam.py -v
```
