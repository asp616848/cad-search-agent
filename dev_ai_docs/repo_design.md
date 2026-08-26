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

We do not run classification or segmentation. We strip the task head and take the graph-level mean-pooled node features after the final message-passing layer as our 512-dim geometric embedding.

This gives us a learned representation of the solid's geometry — not a rule-based feature list.

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
eng-search/
├── dev_ai_docs/          # this folder
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   ├── search.py       # /search/cad, /search/text
│   │   │   ├── library.py      # /parts CRUD, /index
│   │   │   └── mesh.py         # serve glTF for 3D viewer
│   │   ├── core/
│   │   │   ├── graph_builder.py     # lifted from cad-feature-detection
│   │   │   ├── uvnet_embedder.py    # UV-Net forward pass, global pooling
│   │   │   ├── text_embedder.py     # sentence-transformers MiniLM
│   │   │   └── search_index.py      # FAISS + SQLite wrapper
│   │   └── models/
│   │       └── uvnet_weights.pt     # copied from cad-feature-detection
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Search.tsx       # main search page
│   │   │   └── Library.tsx      # browse all indexed parts
│   │   ├── components/
│   │   │   ├── Viewer3D.tsx     # Three.js glTF viewer
│   │   │   ├── ResultCard.tsx   # single search result
│   │   │   └── SearchBar.tsx    # file upload + text input
│   │   └── api/client.ts
│   ├── package.json
│   └── vite.config.ts
└── scripts/
    └── index_library.py    # offline: process demo STEP files → build FAISS index
```
