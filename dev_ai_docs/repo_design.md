# Repository Design — What We Use and Why

## Repos in This Workspace

Five repos were evaluated. Decision rationale for each is in `decisions.log`.

---

### Used: `cad-feature-detection`

**What we lift:** `feature_detector/build_graph.py` and `feature_detector/occ_file_translator.py`

These two files give us the full pipeline:
```
STEP file → pythonOCC → occwl Solid → DGL face-adjacency graph
```
The DGL graph is the input format UV-Net expects. No rewriting needed.

The pre-trained UV-Net weights (`feature_detector/model.tar.gz`) are used as-is.

---

### Used: `UV-Net`

**What we use:** The encoder architecture (`uvnet/models/`) as a reference, and the pre-trained checkpoint.

`GraphEncoder.forward()` returns `(node_emb, graph_emb)`. We use `graph_emb` (128-dim, max-pooled across GNN layers) as the geometric retrieval vector — normalized to unit length. We also run the existing **segmentation head** on `node_emb` to get per-face machining feature labels → a histogram (holes, fillets, pockets, chamfers). The histogram enriches text embeddings and powers feature chips in the UI. No mean-pooling, no head removal — use what the model already provides.

---

### Reference only: `Palmetto`

**What we borrow conceptually:**
- STEP → pythonOCC → mesh → Three.js rendering approach (glTF export)
- FastAPI project structure (routers, model storage pattern)

We do not extend Palmetto or depend on its C++ engine. The OpenCASCADE C++ build is a maintenance burden we don't want in a fresh repo.

---

### Deferred: `AAGNet`

Would add machining feature instance segmentation on top of UV-Net's global embeddings — useful for knowing *which faces* make two parts similar and for richer metadata. Requires its own gAAG preprocessing pipeline and pre-trained weights not in the repo.

Slot for v2: run AAGNet on indexed parts to enrich metadata labels → better text embeddings.

---

### Not used: `VLM-CADFeatureRecognition`

Image-based. Requires rendering STEP → multi-view images before inference. Adds a rendering pipeline dependency with no architectural advantage over UV-Net for similarity search. Potential v2 addition as a complementary search modality ("find parts that look like this image").

---

## Our Repo Structure

```
cad-search-agent/
├── dev_ai_docs/          # this folder
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py             # all weights/thresholds sourced from env vars
│   │   ├── api/
│   │   │   ├── search.py         # /search/cad, /search/text
│   │   │   ├── library.py        # /library/index, /library/add-query, /parts
│   │   │   ├── mesh.py           # serve glTF + thumbnails
│   │   │   └── health.py         # /health: model loaded, index size, last indexed
│   │   ├── core/
│   │   │   ├── graph_builder.py  # lifted from cad-feature-detection (occwl → DGL)
│   │   │   ├── uvnet_embedder.py # graph_emb (128-dim) + seg head → histogram
│   │   │   ├── occ_stats.py      # face/edge count, volume, bbox via pythonOCC
│   │   │   ├── text_embedder.py  # MiniLM on enriched metadata string
│   │   │   ├── search_index.py   # two FAISS indexes + score fusion
│   │   │   └── llm_adapter.py    # anthropic | gemini | template fallback + cache
│   │   ├── data/                 # geo_index.faiss, text_index.faiss, parts.db
│   │   └── models/
│   │       └── uvnet_weights.pt  # copied from cad-feature-detection
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Search.tsx        # upload + results + side-by-side viewer
│   │   │   └── Library.tsx       # thumbnail grid + facet filters
│   │   ├── components/
│   │   │   ├── Viewer3D.tsx      # Three.js glTF, side-by-side mode
│   │   │   ├── ResultCard.tsx    # dual score bars + feature chips + badges
│   │   │   ├── CostingPrior.tsx  # cost band + supplier hint + DFM warning + copy
│   │   │   └── SearchBar.tsx     # file upload + text + stage spinner
│   │   └── api/client.ts
│   ├── package.json
│   └── vite.config.ts
├── scripts/
│   ├── index_library.py          # STEP dir → FAISS + SQLite + glTF + thumbnails
│   └── eval_retrieval.py         # 8 labeled queries → Recall@3 report
└── data/
    ├── demo_steps/               # 30 STEP files committed (not cloned on demo day)
    └── demo_metadata.json        # metadata for all 30 parts
```
