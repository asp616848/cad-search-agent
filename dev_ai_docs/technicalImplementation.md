# Technical Implementation

## Stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI + Python 3.11 | Familiar, async, easy to run locally |
| CAD parsing | pythonOCC + occwl | STEP → B-rep topology, required by UV-Net |
| ML / Geometry | UV-Net (PyTorch + DGL, CPU) | Pre-trained, real learned B-rep embeddings |
| Text embeddings | sentence-transformers MiniLM-L6-v2 | Local, no API call, fast on CPU |
| LLM | Claude Sonnet (Anthropic API) | "Why similar" explanations only |
| Vector search | FAISS (flat L2) | Zero infra, works locally |
| Metadata store | SQLite | No server needed |
| Frontend | React + Vite + Tailwind | Familiar |
| 3D viewer | Three.js + react-three-fiber | STEP → glTF (via pythonOCC) → render |

---

## Data Flow

### A. Offline Indexing (run once to build demo library)

```
STEP file
  → occ_file_translator: load into pythonOCC topology
  → occwl: wrap as Solid → extract face-adjacency graph
  → DGL graph (UV-grid node/edge features)
  → UV-Net encoder (forward pass, CPU)
  → global mean pool over final node features
  → 512-dim geometric embedding  ──┐
                                    ├─ normalize → concat → FAISS index
Part metadata (name, process,       │
  material, notes as text string)   │
  → MiniLM → 384-dim text emb ─────┘

Metadata row → SQLite (id, name, material, process, cost, notes, mesh_path)
Embedding row → numpy array, saved alongside FAISS index
```

### B. Real-Time Query (user uploads STEP)

```
Upload STEP
  → same CAD → DGL → UV-Net → geo_emb
  → MiniLM on any text the user typed → text_emb
  → combine: 0.7 * geo_emb + 0.3 * text_emb (normalized, concatenated)
  → FAISS.search(k=5)
  → fetch metadata for top-5 from SQLite
  → POST to Claude: "Given this query part's features and these results, explain similarity"
  → return: [{part, score, claude_explanation, mesh_url}]
```

### C. Text-Only Query

```
User types "M6 tapped aluminum bracket"
  → MiniLM embed
  → FAISS.search on text subspace only (just the 384-dim slice)
  → same result format
```

---

## UV-Net Embedding Extraction — Key Detail

UV-Net was trained for face-level segmentation (classifying each face as a machining feature type). We repurpose it as a geometry encoder:

```python
# uvnet_embedder.py (simplified)
model.load_state_dict(torch.load("models/uvnet_weights.pt", map_location="cpu"))
model.eval()

with torch.no_grad():
    # run graph through UV-Net's GNN encoder only (no task head)
    node_feats = model.graph_encoder(graph)     # (num_faces, 512)
    solid_emb = node_feats.mean(dim=0)          # (512,) — global pooling
    solid_emb = F.normalize(solid_emb, dim=0)  # unit vector for cosine-stable L2
```

This is standard transfer learning practice. The encoder learns geometry; we skip the classification head.

---

## API Endpoints

```
POST /api/search/cad          Upload STEP → top-k similar parts
POST /api/search/text         Text query → top-k similar parts
POST /api/library/index       Add a new part to the library (STEP + metadata)
GET  /api/library/parts       List all indexed parts
GET  /api/library/parts/{id}  Part detail + mesh URL
GET  /api/mesh/{id}           Serve glTF mesh for 3D viewer
```

---

## Frontend — Two Views

**Search page**
- Left: drag-drop STEP upload zone + optional text input
- Right: results grid — cards showing part name, similarity %, manufacturing info, mini 3D preview
- Click result → expand with full 3D viewer + Claude explanation

**Library page**
- Grid of all indexed parts (thumbnails, name, process, material)
- Click any part → detail view + "find similar" shortcut

---

## STEP → glTF for 3D Viewer

pythonOCC can tessellate a STEP solid and export glTF without the Palmetto C++ engine:

```python
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Extend.DataExchange import read_step_file
from OCC.Core.RWGltf import RWGltf_CafWriter
# tessellate + export — ~10 lines, no C++ build required
```

---

## Production Notes (for interview — what comes after POC)

- FAISS → Qdrant (supports filtered search by material/process)
- SQLite → Postgres
- UV-Net fine-tuned on Pre6's actual historical parts (the pre-trained weights generalize but miss domain-specific geometry)
- MiniLM → OpenAI `text-embedding-3-large` for richer semantic matching
- Embeddings pre-computed on part upload, not on search query
- Assembly support: embed each body in a multi-body STEP separately, store as a set
