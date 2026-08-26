# Testing & Demo Guide

## Getting STEP Files (You Have No CAD Experience — That's Fine)

### Source 1: Already on disk (start here)
```
/Users/abhijeet/Documents/GitHub/pre6/cad-feature-detection/files/
/Users/abhijeet/Documents/GitHub/pre6/AAGNet/examples/
```
These are validated STEP files already confirmed working with pythonOCC + occwl. Use these to verify the UV-Net pipeline before touching anything else.

Note: Palmetto `examples/test-models/` is listed in the README but is often empty. Don't rely on it.

### Source 2: MFCAD Dataset (for the 30-part demo library)
- Repo: https://github.com/hducg/MFCAD
- 24 machining feature types, ~3000 STEP files, organized by feature class
- Download:
```bash
git clone https://github.com/hducg/MFCAD /tmp/mfcad
ls /tmp/mfcad/  # see feature categories
```

**After cloning, copy 30 selected files to `data/demo_steps/` inside the repo.** The demo must not depend on a git clone on interview morning.

### Source 3: MFInstSeg (if MFCAD isn't diverse enough)
- Google Drive link in `AAGNet/README.md` — 60k STEP files
- Use a subset of 10–15 only

---

## Smoke Order — Do Not Skip

Run in this exact order. Do not open Vite until step 2 passes.

**Step 1: Isolated embed test**
```bash
cd cad-search-agent/backend
python -c "
from app.core.graph_builder import step_to_graph
from app.core.uvnet_embedder import embed
graph = step_to_graph('path/to/any_test.step')
geo_vec, histogram = embed(graph)
print('geo_vec shape:', geo_vec.shape)    # expect (128,)
print('histogram:', histogram)             # e.g. {'hole': 4, 'fillet': 6}
import torch; print('norm:', torch.norm(geo_vec).item())  # expect ~1.0
"
```

**Step 2: Retrieval eval (run before building UI)**
```bash
python scripts/eval_retrieval.py
```
This prints Recall@3 and ranked names for 8 labeled queries. If the near-duplicate is not ranked #1, the demo is broken — fix the embedding before touching frontend.

**Step 3: Full index build**
```bash
python scripts/index_library.py \
  --step_dir data/demo_steps/ \
  --metadata data/demo_metadata.json \
  --output backend/app/data/
```
Script prints a library card on completion: count by process/material, cost range, list of failed files. Failed STEP files must not silently vanish.

**Step 4: Backend health check**
```bash
uvicorn app.main:app --reload --port 8000
curl http://localhost:8000/api/health
# expect: {"status": "ok", "model_loaded": true, "index_size": 30, "last_indexed": "..."}
```

**Step 5: UI**
```bash
cd frontend && npm run dev
```

---

## Building the 30-Part Demo Library

Pick parts that tell a clear story. Variety matters more than quantity.

| Category | Count | What to look for in MFCAD |
|---|---|---|
| Brackets | 6 | `L_bracket`, `bracket` in filename/folder |
| Flat plates | 5 | `plate`, `flange` |
| Cylinders / shafts | 5 | `shaft`, `cylinder` |
| Complex / intersecting features | 4 | folders with multiple feature types combined |
| Near-duplicate pair | 2 | Two files from the same MFCAD class that are geometrically close — UV-Net should score them ≥ 0.90 |
| Weak-match bait | 4 | Deliberately different geometry (shaft vs bracket) — low scores confirm the system works |
| Simple block / cube | 2 | Expect clustering — good for showing the Weak-match badge and talking about the UV-Net limitation honestly |
| One "ugly" part | 1 | Multi-body or heavy spline — shows the error state on purpose |
| Spare | 1 | Buffer for anything that fails OCCwl |

Confirm your near-duplicate pair with `eval_retrieval.py` before writing the demo script.

---

## `eval_retrieval.py` — Golden Retrieval Script

Write 8 labeled queries before the demo, covering:

| Query | What must appear in top-3 | Score gate |
|---|---|---|
| Near-duplicate STEP | its pair | geo ≥ 0.90 |
| Any bracket STEP | other brackets, not shafts | — |
| Text: "M6 aluminum counterbore" | parts with those tokens in metadata | — |
| Shaft STEP | shafts | — |
| Plate STEP | plates | — |
| Text: "CNC milling Acme" | parts with Acme supplier | — |
| Cube / block STEP | assert top score < 0.80, show Weak-match state | — |
| First-of-kind (unusual geometry not in library) | assert top score < 0.55 | — |

Print Recall@3 per query and overall. Target: ≥ 6/8 queries passing before interview.

---

## `demo_metadata.json` — Write as Search Documents

Each entry must contain tokens an engineer would actually search. The file powers both MiniLM text embeddings and the Costing-prior rail.

```json
[
  {
    "filename": "bracket_001.step",
    "name": "L-Bracket Rev3",
    "material": "Aluminum 6061",
    "process": "CNC Milling",
    "cost": 420,
    "supplier": "Acme Machining",
    "notes": "M6 tapped holes, tight tolerance on mounting face, anodized finish",
    "known_issues": "thin wall near inner pocket caused chatter at 3mm depth",
    "ppap_notes": "PPAP Level 2 submitted 2024-03. Approved."
  }
]
```

Rules:
- Put `M6`, `counterbore`, `6061`, `anodize`, `PPAP`, `thin wall` as appropriate — these are text search targets
- Make 2–3 parts share a supplier (Acme Machining) so "supplier hint" fires as a live result
- Make 2–3 parts share a `known_issues` tag (e.g. `thin wall`) that overlaps with a histogram label — so DFM warning fires in the prior rail
- Include cost for all 30 parts so cost band works for every query

---

## Demo Script (for interview)

**Step 1 — CAD similarity search**
Upload a bracket STEP from outside the library. System returns similar brackets. Say: "The similarity is computed from UV-Net B-rep embeddings — the model encodes the face-adjacency graph of the solid, not pixel renders."

**Step 2 — Dual score bars**
Point to the Geometry and Text bars on a result card. Say: "Geometry and text are kept as separate signals. A text-only query — like typing a part name — skips UV-Net entirely and searches the metadata index."

**Step 3 — Near-duplicate detection**
Upload the near-duplicate STEP. Top result should be ≥ 0.90 geo score with the Near-duplicate badge. Say: "At 0.95+ we flag the part as likely already quoted — prevents redundant engineering work."

**Step 4 — Costing prior**
Click the top result. Show the prior rail: cost, cost band, supplier, known issues. Click "Copy context for Costing." Say: "This blob becomes the grounding context for Pre6 Costing — estimated cost is not a blank-slate LLM guess, it's anchored to what we actually charged for a geometrically similar part."

**Step 5 — Weak match / first-of-kind**
Upload a shaft when the library has brackets. Low scores, Weak-match badges. Say: "The system correctly says 'nothing similar found' — the right answer for a first-of-kind part is to start fresh with Costing, not return a misleading result."

**Optional — knowledge loop**
After step 1, click "Add to library." Re-search the same STEP. It now appears as hit #1. Say: "Every project added to Pre6 enriches the knowledge base — switching cost compounds over time."

---

## Before the Interview

- [ ] `eval_retrieval.py` passes ≥ 6/8 queries
- [ ] 30 STEP files committed to `data/demo_steps/` (not git-cloned on interview morning)
- [ ] `demo_metadata.json` written with all 30 entries — every cost, supplier, known_issues filled
- [ ] `/api/health` returns `index_size: 30`
- [ ] Screen-record a 90-second run of all 5 demo steps as backup (wifi / API issues happen)
- [ ] Know which part triggers the spline error — show it on purpose, not by accident

---

## Known Issues to Call Out Proactively

- **Simple geometry (cubes, blocks):** UV-Net embeddings cluster near 0.8 for everything. Show the Weak-match badge; say "the model is most discriminative on prismatic machined parts."
- **Multi-body STEP:** Only first solid embedded in v1. Frontend shows a banner; say "assembly support is the next arch change."
- **Spline/freeform surfaces:** occwl may fail. Frontend shows engineer-friendly error. Keep one such file in the demo set to show it on purpose.
