# Testing & Demo Guide

## Getting STEP Files (You Have No CAD Experience — That's Fine)

### Source 1: MFCAD Dataset (best for demo)
- Repo: https://github.com/hducg/MFCAD
- 24 machining feature types (holes, pockets, slots, chamfers, etc.)
- ~3,000 STEP files, already organized by feature type
- Download: clone the repo or grab specific folders
- Files are named by feature type — easy to pick diverse parts

```bash
git clone https://github.com/hducg/MFCAD /tmp/mfcad
ls /tmp/mfcad/  # see feature categories
```

### Source 2: Palmetto examples (already on disk)
```
/Users/abhijeet/Documents/GitHub/pre6/Palmetto/examples/test-models/
/Users/abhijeet/Documents/GitHub/pre6/Palmetto/examples/sample-models/
```
These are validated STEP files that already work with pythonOCC — start testing with these.

### Source 3: AAGNet examples (already on disk)
```
/Users/abhijeet/Documents/GitHub/pre6/AAGNet/examples/
```
3 STEP files included for visualization testing.

### Source 4: cad-feature-detection
```
/Users/abhijeet/Documents/GitHub/pre6/cad-feature-detection/files/
```
Existing test STEP files already validated against UV-Net pipeline.

---

## Building the Demo Library (30 Parts)

Pick parts that tell a clear story when searched. Suggested mix:

| Category | Count | Feature types | Why |
|---|---|---|---|
| Brackets | 6 | holes, chamfers, fillets | common, easy to show "similar bracket" match |
| Plates | 5 | pockets, countersinks | flat parts, clearly different from brackets |
| Shafts / cylinders | 5 | turned features, grooves | different geometry class |
| Complex / intersecting | 4 | multiple feature types | shows model handles hard cases |
| Near-duplicates (2 pairs) | 4 | same part, slight variation | shows similarity scoring working |
| Intentionally different | 6 | wildly different geometry | shows low similarity scores are correct |

---

## Pre-Processing the Library (run offline once)

```bash
cd eng-search
python scripts/index_library.py \
  --step_dir /tmp/mfcad/StepFiles \
  --metadata scripts/demo_metadata.json \
  --output backend/app/data/
```

`demo_metadata.json` is a manually written file that gives each STEP file fake-but-realistic metadata:

```json
[
  {
    "filename": "bracket_001.step",
    "name": "L-Bracket Rev3",
    "material": "Aluminum 6061",
    "process": "CNC Milling",
    "cost": 420,
    "supplier": "Acme Machining",
    "notes": "M6 tapped holes, tight tolerance on mounting face"
  },
  ...
]
```

This metadata is what Claude uses to generate the "why similar" explanation and what appears on result cards. Make it realistic enough to be convincing in a demo.

---

## Running the App

```bash
# Terminal 1 — backend
cd eng-search/backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd eng-search/frontend
npm run dev
# → http://localhost:5173
```

---

## Demo Script (for interview)

**Step 1: Show the search works**
- Upload one of the MFCAD bracket STEP files as a "new RFQ"
- System returns top-3 similar brackets with similarity scores
- Point out: "The similarity is computed from UV-Net B-rep embeddings, not pixel matching"

**Step 2: Show text search**
- Type "aluminum plate with M6 counterbored holes"
- Returns relevant flat parts — even without uploading a CAD file

**Step 3: Show the near-duplicate pair**
- Upload the slightly-modified version of a part already in the library
- Similarity score should be >0.95 — point out duplicate detection use case

**Step 4: Show the business story**
- Click a result → show its metadata (cost, supplier, process, PPAP notes)
- "This retrieved context feeds directly into Pre6 Costing as a prior — the estimate is grounded in what we actually charged for a similar part"

**Step 5: Show a non-match**
- Upload a shaft/cylinder when the library has mostly brackets
- Low similarity scores across the board — system correctly says "nothing similar found"

---

## Verifying UV-Net Works Before Full App

Test the embedding pipeline in isolation:

```bash
cd eng-search/backend
python -c "
from app.core.graph_builder import step_to_graph
from app.core.uvnet_embedder import embed
import torch

graph = step_to_graph('path/to/test.step')
emb = embed(graph)
print('Embedding shape:', emb.shape)   # expect (512,)
print('Norm:', torch.norm(emb).item()) # expect ~1.0
"
```

If this runs without error, the ML pipeline is working.

---

## Known Issues to Test For

- Multi-body STEP files: UV-Net embeds one solid at a time. If STEP has assembly, only the first body is embedded. Log a warning.
- Very simple geometry (e.g., a cube): UV-Net embeddings are less discriminative — similarity scores cluster near 0.8 for everything. Expected; note it in the demo.
- STEP files with spline surfaces: occwl may fail on complex freeform geometry. Catch the exception and return a clear error to the user.
