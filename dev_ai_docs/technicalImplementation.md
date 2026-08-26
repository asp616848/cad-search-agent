# Technical Implementation

## Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Python 3.11 | Familiar, async, easy to run locally |
| CAD parsing | pythonOCC + occwl | STEP → B-rep topology, required by UV-Net |
| ML / Geometry | UV-Net (PyTorch + DGL, CPU) | Pre-trained, real learned B-rep embeddings |
| Text embeddings | sentence-transformers MiniLM-L6-v2 | Local, no API call, fast on CPU |
| LLM (optional) | Claude or Gemini behind one adapter | "Why similar" explanations; degrades to template if unavailable |
| Vector search | Two FAISS flat indexes (geo + text) | Score fusion; separate indexes allow text-only and CAD-only paths |
| Metadata store | SQLite | No server needed |
| Frontend | React + Vite + Tailwind | Familiar |
| 3D viewer | Three.js + react-three-fiber | STEP → glTF (via pythonOCC) → render |

---

## UV-Net Embedding — What the Model Actually Returns

UV-Net's `GraphEncoder.forward()` returns `(node_emb, graph_emb)`.

- `graph_emb` — **128-dim**, graph-level max-pool across GNN layers. This is what the model was trained to use. Use this as the geometric retrieval vector.
- `node_emb` — per-face embeddings. Feed these into the existing **segmentation head** (already in the checkpoint) to get per-face machining feature labels.

Do not invent a mean-pool over node features. The model did that work; use `graph_emb`.

```python
# uvnet_embedder.py
model.load_state_dict(torch.load("models/uvnet_weights.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    node_emb, graph_emb = model.graph_encoder(graph)   # (N, d), (128,)
    geo_vec = F.normalize(graph_emb, dim=0)             # unit vector

    face_logits = model.seg_head(node_emb)              # (N, num_classes)
    face_labels = face_logits.argmax(dim=1)             # per-face label

# Feature histogram from face labels
histogram = {label: count for label, count in zip(*face_labels.unique(return_counts=True))}
# e.g. {"hole": 4, "fillet": 6, "pocket": 1, "chamfer": 0, ...}
```

The histogram is stored in SQLite alongside metadata and used for:
- Feature chips on result cards (overlap = green, one-sided = muted)
- Enriching the text string fed to MiniLM ("4 holes, 6 fillets, 1 pocket — M6 tapped aluminum bracket...")
- Optional Jaccard overlap as a third score signal

---

## Geometric Fingerprints (no extra ML)

At index time, pythonOCC already has the solid. Extract and store:

```python
# occ_stats.py
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

props = GProp_GProps()
brepgprop.VolumeProperties(shape, props)
volume = props.Mass()
brepgprop.SurfaceProperties(shape, props)
surface_area = props.Mass()

bbox = Bnd_Box()
brepbndlib.Add(shape, bbox)
xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
dims = (xmax-xmin, ymax-ymin, zmax-zmin)
solidity = volume / (dims[0] * dims[1] * dims[2])  # 0–1
```

Stored fields: `face_count`, `edge_count`, `volume`, `surface_area`, `bbox_x/y/z`, `solidity`.

Use as tie-breakers when UV-Net scores cluster (e.g. simple blocks) and as explanation fodder ("similar bounding box, similar face count").

---

## Search Architecture — Two Indexes + Score Fusion

**Two separate FAISS flat indexes:**
- `geo_index`: 128-dim unit vectors (UV-Net `graph_emb`)
- `text_index`: 384-dim unit vectors (MiniLM on metadata string)

```python
# search_index.py
cad_score  = cosine(q_geo,  library_geo)    # scalar per part
text_score = cosine(q_text, library_text)   # scalar per part, 0 if no text query

# Weights in config.py, not hardcoded
final = GEO_WEIGHT * cad_score + TEXT_WEIGHT * text_score  # default 0.7 / 0.3
```

| Query type | Behavior |
|---|---|
| STEP only | `final = cad_score`; text_score = 0 |
| Text only | `final = text_score`; skip UV-Net entirely |
| Both | fusion with configured weights |

Return both scores to the frontend — "Geometry 82% · Text 41%" is the single best interview talking point.

---

## Data Flow

### A. Offline Indexing (`scripts/index_library.py`, run once)

```
STEP file
  → pythonOCC: load topology, extract OCC stats (face/edge count, volume, bbox)
  → occwl: wrap as Solid → DGL face-adjacency graph
  → UV-Net: graph_emb (128-dim, normalized) + face segmentation → histogram
  → pythonOCC: tessellate → glTF mesh + thumbnail PNG
  → MiniLM: embed(name + material + process + feature histogram + notes) → 384-dim text_vec

Write to:
  geo_index.faiss    (128-dim vectors)
  text_index.faiss   (384-dim vectors)
  parts.db           (SQLite: all metadata + OCC stats + histogram + paths)
  meshes/{id}.glb
  thumbnails/{id}.png
```

After indexing, script prints a library card: count by process/material, cost range, failed files.

### B. Real-Time CAD Query

```
Upload STEP
  → same OCC + occwl + UV-Net path → q_geo (128-dim)
  → MiniLM on any text typed → q_text (384-dim, or zeros)
  → FAISS search both indexes → score fusion → top-5
  → SQLite fetch metadata for top-5
  → (if LLM_API_KEY) Claude/Gemini: 2–3 sentence explanation (cached by query+result hash)
  → (else) template from scores + overlapping histogram tags
  → log: graph_ms, uvnet_ms, faiss_ms on response
```

### C. Text-Only Query

```
Text input
  → MiniLM → q_text (384-dim)
  → FAISS search text_index only → top-5
  → same result format
```

---

## Index-Time vs Query-Time (explicit)

| When | Work done |
|---|---|
| `index_library.py` (offline) | OCC stats + UV-Net geo_vec + seg histogram + MiniLM text_vec + glTF + thumbnail → written to FAISS + SQLite |
| Upload query (real-time) | Same geo + text pipeline for the **query vector only**; library is never re-embedded |
| Text query (real-time) | MiniLM only; UV-Net and pythonOCC not touched |

---

## LLM — Optional, Never on Critical Path

```python
# llm_adapter.py
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none")  # "anthropic" | "gemini" | "none"

def explain(query_meta, result_meta, scores) -> str:
    if LLM_PROVIDER == "anthropic":
        return _anthropic_explain(...)
    elif LLM_PROVIDER == "gemini":
        return _gemini_explain(...)
    else:
        return _template_explain(query_meta, result_meta, scores)

def _template_explain(q, r, scores):
    tags = overlapping_histogram_tags(q, r)
    return (
        f"Geometry match {scores['geo']:.0%}. "
        f"Shared features: {', '.join(tags) or 'none detected'}. "
        f"{r['name']} used {r['process']} at ${r['cost']}."
    )
```

Cache explanations in SQLite: `(query_hash, result_id) → explanation`. Repeat demo clicks do not re-bill the API.

---

## API Endpoints

```
GET  /api/health                   Model loaded, index size, last index time
POST /api/search/cad               Upload STEP → top-k results + dual scores + explanation
POST /api/search/text              Text query → top-k results
POST /api/library/index            Add a part to the library (STEP + metadata JSON)
POST /api/library/add-query/{id}   Add the last query result to the library (knowledge loop)
GET  /api/library/parts            List all indexed parts (with facet filter: material, process)
GET  /api/library/parts/{id}       Part detail + mesh URL + histogram
GET  /api/mesh/{id}                Serve glTF
GET  /api/thumbnail/{id}           Serve thumbnail PNG
```

---

## Frontend — Views and Key Components

### Search Page
- **Left (sticky):** drag-drop STEP upload zone + optional text input + upload stage text ("Reading B-rep → UV-Net → Searching")
- **Right:** ranked result cards

**Result card:**
- Part name, material, process, cost
- Dual score bar: "Geometry 82% · Text 41%"
- Feature chips: overlapping = green, one-sided = muted
- Badge: **Near-duplicate** if geo ≥ 0.95 · **Weak match** if top score < 0.55
- LLM explanation paragraph below (loads async; card is useful without it)

**Selected result expands to:**
- Side-by-side 3D viewer (query | result, same camera)
- Costing Prior rail (right side):
  - Cost from this part
  - Cost band from top-3 neighbors with geo ≥ 0.7: "Historical quotes: $380–$510 (n=3)"
  - Process, material, supplier
  - Known issues / PPAP notes
  - **Copy context for Costing** button → copies plaintext blob to clipboard

### Library Page
- Thumbnail grid of all indexed parts
- Facet chips: material / process (SQLite WHERE, no vector math)
- "Find similar" button on any part (reuses stored `geo_emb`, no UV-Net re-run)

### Upload UX
- Accept `.step` / `.stp` / `.STEP`; reject all else with a one-liner
- 25 MB cap
- Spinner with stage labels: "Reading B-rep → Building graph → UV-Net → Searching"
- Last query stays on screen when results arrive

### Error States (sound like an engineer)
- Spline/occwl failure: "This STEP has freeform surfaces we can't convert to a B-rep graph. Try a prismatic solid (milled or turned part)."
- Nothing similar: "No close geometry in the library — expected for a first-of-kind part. Start a fresh Costing run."
- Weak match: banner on cards with score < 0.55

---

## Business Extensions — Stubbed from Metadata (no extra ML)

All four extensions from `Business.md` are live in v1 as UI stubs, not roadmap slides:

| Extension | Implementation |
|---|---|
| **Duplicate gate** | If geo cosine ≥ 0.95: banner "This looks like an existing part." Link the library entry. |
| **Cost band** | top-k with geo ≥ 0.7 → min/median/max of `cost` field → "Historical quotes: $380–$510 (n=3)" |
| **Supplier hint** | Count `supplier` in top-3 → "2 of 3 closest parts went to Acme Machining" |
| **DFM from history** | If `known_issues` tag overlaps with query histogram label → highlight in prior rail |

All four require only SQLite reads on the top-k result set. No additional models.

---

## Configuration (`config.py`)

```python
GEO_WEIGHT    = float(os.getenv("GEO_WEIGHT", "0.7"))
TEXT_WEIGHT   = float(os.getenv("TEXT_WEIGHT", "0.3"))
TOP_K         = int(os.getenv("TOP_K", "5"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.95"))
WEAK_THRESHOLD= float(os.getenv("WEAK_THRESHOLD", "0.55"))
LLM_PROVIDER  = os.getenv("LLM_PROVIDER", "none")   # anthropic | gemini | none
LLM_API_KEY   = os.getenv("LLM_API_KEY", "")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
```

---

## Production Notes (for interview)

- FAISS flat → Qdrant (supports filtered search by material/process, not just vector distance)
- SQLite → Postgres
- UV-Net fine-tuned on Pre6's actual historical parts (pre-trained weights generalize; domain-specific geometry improves recall)
- MiniLM → OpenAI `text-embedding-3-large` for richer semantic matching
- GEO_WEIGHT / TEXT_WEIGHT learned from click data (search log table: `searches(id, ts, mode, top_ids, clicked_id)`)
- Timestamp + clicked result already tracked in POC SQLite; production trains a small ranker on it
