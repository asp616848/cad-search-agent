# Implementation Plan — Phased, Gated, Self-Improving

Read [`../PRD.md`](../PRD.md) first for the philosophy. This file is the operational plan:
what to build in each phase, the **exact command that proves it works**, and how to loop when a
gate fails.

**Rule:** a phase is done only when its gate command exits green. Do not start phase N+1 with a
red phase N. Commit at every green gate.

Legend: 🎯 = deliverable · ✅ = gate (must pass) · 🔁 = common failure → knob to turn.

---

## Phase 0 — Skeleton, config, CI shell

🎯 Deliverables
- `backend/` Python package: `app/`, `app/config.py`, empty `app/core/`, `app/api/`.
- `backend/requirements.txt` (fastapi, uvicorn, faiss-cpu, sentence-transformers, numpy, pytest,
  ruff, black; pythonOCC + DGL + torch pinned but installed locally, see note).
- `backend/tests/` with `unit/`, `integration/`, `api/`, `fixtures/`.
- `frontend/` Vite + React + TS shell (`npm create vite`), Tailwind wired.
- `config.py` reads every knob from env with defaults (list in
  [`technicalImplementation.md` → Configuration](technicalImplementation.md)).
- `.github/workflows/ci.yml` with lint + empty pytest jobs (see §CI rollout).
- `data/`, `data/demo_steps/` (empty, `.gitkeep`), `.gitignore` for `*.faiss`, `parts.db`, `.venv`.

✅ Gate
```bash
cd backend && ruff check . && black --check . && pytest -q         # collects, 0 failures
cd ../frontend && npm run build                                     # builds clean
```
CI green on the empty suite.

🔁 If pythonOCC/DGL install fights you on macOS: pin `pythonocc-core` via conda, `dgl` CPU wheel;
record the working versions in `requirements.txt` comments. This is the one env-risk phase — solve
it now, not at Phase 4.

---

## Phase 1 — STEP → graph → geo embedding + histogram

🎯 Deliverables
- `app/core/graph_builder.py` — lift `build_graph.py` + `occ_file_translator.py` from
  `cad-feature-detection` (see [`repo_design.md`](repo_design.md)). `step_to_graph(path) -> DGLGraph`.
- `app/core/uvnet_embedder.py` — load `models/uvnet_weights.pt`; `embed(graph) -> (geo_vec[128],
  histogram: dict)`. Uses `graph_emb` + segmentation head (see
  [`technicalImplementation.md` → UV-Net Embedding](technicalImplementation.md)).
- Copy `uvnet_weights.pt` from `cad-feature-detection/feature_detector/model.tar.gz`.
- 3 STEP fixtures in `backend/tests/fixtures/` (from `cad-feature-detection/files/` or
  `AAGNet/examples/`).

✅ Gate — `backend/tests/integration/test_embedder.py`
```python
def test_geo_vec_shape_and_norm():
    g = step_to_graph(FIXTURE)          # bracket.step
    vec, hist = embed(g)
    assert vec.shape == (128,)
    assert abs(float(torch.norm(vec)) - 1.0) < 1e-3
    assert sum(hist.values()) > 0        # some faces got labeled

def test_two_different_parts_differ():
    v1, _ = embed(step_to_graph(BRACKET))
    v2, _ = embed(step_to_graph(SHAFT))
    assert cosine(v1, v2) < 0.999        # not degenerate/identical
```
```bash
pytest backend/tests/integration/test_embedder.py -q
```

🔁 Wrong shape → you took `node_emb` not `graph_emb`. Norm ≠ 1 → forgot `F.normalize`. Empty
histogram → seg head not wired / wrong checkpoint key. All identical vectors → graph features not
populated (UV-grids missing).

---

## Phase 2 — OCC stats + text embedder

🎯 Deliverables
- `app/core/occ_stats.py` — `stats(shape) -> {face_count, edge_count, volume, surface_area,
  bbox_xyz, solidity}` (code sketch in [`technicalImplementation.md`](technicalImplementation.md)).
- `app/core/text_embedder.py` — MiniLM wrapper; `embed_text(str) -> vec[384]` (normalized).

✅ Gate — `backend/tests/unit/`
```python
def test_occ_stats_ranges():
    s = stats(load_shape(BRACKET))
    assert s["face_count"] > 0 and s["volume"] > 0
    assert 0 < s["solidity"] <= 1.0

def test_text_vec_shape():
    v = embed_text("M6 aluminum bracket")
    assert v.shape == (384,) and abs(np.linalg.norm(v) - 1) < 1e-3
```

🔁 solidity > 1 → bbox/volume swapped or bbox includes gap. Text norm ≠ 1 → normalize flag off.

---

## Phase 3 — Dual FAISS index + fusion + SQLite

🎯 Deliverables
- `app/core/search_index.py`:
  - two `faiss.IndexFlatIP` (geo 128, text 384) over unit vectors (inner product == cosine).
  - SQLite schema: `parts(id, name, material, process, cost, supplier, notes, known_issues,
    ppap_notes, histogram_json, occ_stats_json, mesh_path, thumb_path, geo_vec_blob)`.
  - `add(part)`, `search_cad(geo_vec, k)`, `search_text(text_vec, k)`,
    `fuse(cad_scores, text_scores) -> ranked` using `GEO_WEIGHT/TEXT_WEIGHT` from config.

✅ Gate — `backend/tests/integration/test_search.py`
```python
def test_part_finds_itself():
    idx = build_index([BRACKET, SHAFT, PLATE])   # helper indexes fixtures
    hits = idx.search_cad(embed(step_to_graph(BRACKET))[0], k=3)
    assert hits[0].id == BRACKET_ID and hits[0].geo_score > 0.99

def test_fusion_math():
    r = fuse({A:0.8, B:0.2}, {A:0.0, B:1.0})     # weights 0.7/0.3
    assert abs(r[A] - 0.56) < 1e-6 and abs(r[B] - 0.44) < 1e-6
```

🔁 Self-search not #1 → not normalizing before add, or L2 vs IP mismatch. Fusion off →
weights not read from config.

---

## Phase 4 — Library indexer + retrieval metrics (the ML gate)

🎯 Deliverables
- `scripts/index_library.py` — STEP dir + `demo_metadata.json` → FAISS + SQLite + glTF +
  thumbnails; prints library card (counts, cost range, failed files). See
  [`testing_demo.md` → Pre-Processing](testing_demo.md).
- `scripts/eval_retrieval.py` — reads `data/eval_labels.json`, runs queries, writes
  `data/eval_report.json`, prints Recall@1/3/5, MRR, per-query rank, latencies.
- `data/eval_labels.json` — 8 labeled queries (spec in [`testing_demo.md`](testing_demo.md)).
- Build the near-duplicate pair and confirm it here.

✅ Gate
```bash
python scripts/index_library.py --step_dir backend/tests/fixtures --metadata data/eval_meta.json --output backend/app/data
python scripts/eval_retrieval.py           # exits non-zero if targets in PRD §6 not met
```
Gate = `eval_retrieval.py` returns 0 (Recall@3 ≥ 0.75, MRR ≥ 0.70, near-dup rank == 1).
> Note: run on the fixture/eval subset in dev; run again on the full 30-part library before the demo.

🔁 Recall low → (a) enrich histogram into the text string; (b) adjust `GEO_WEIGHT`; (c) check a
mislabeled query in `eval_labels.json`. **Change one knob, re-run, log it in `decisions.log`.**
This is the core self-improving loop — see [`../PRD.md` §7](../PRD.md).

---

## Phase 5 — FastAPI endpoints

🎯 Deliverables — `app/api/`: `health.py`, `search.py`, `library.py` (endpoints listed in
[`technicalImplementation.md` → API Endpoints](technicalImplementation.md)). Return dual scores,
histogram overlap, badges, latencies.

✅ Gate — `backend/tests/api/test_api.py` (FastAPI `TestClient`, seeded fixture index)
```python
def test_health(): assert client.get("/api/health").json()["model_loaded"] is True
def test_search_cad_returns_ranked():
    r = client.post("/api/search/cad", files={"file": open(BRACKET,"rb")}).json()
    assert r["results"][0]["geo_score"] > r["results"][-1]["geo_score"]
    assert {"geo_score","text_score","badge"} <= r["results"][0].keys()
def test_text_only_skips_uvnet():
    r = client.post("/api/search/text", json={"q":"aluminum bracket"}).json()
    assert len(r["results"]) > 0
```

🔁 500 on upload → temp-file handling / size cap. Missing keys → response model not updated.

---

## Phase 6 — Mesh + thumbnails

🎯 Deliverables — `app/api/mesh.py`: `GET /api/mesh/{id}` (glTF), `GET /api/thumbnail/{id}` (PNG).
glTF export via pythonOCC (`RWGltf_CafWriter`), thumbnails rendered at index time.

✅ Gate
```python
def test_mesh_bytes_valid():
    b = client.get(f"/api/mesh/{BRACKET_ID}").content
    assert b[:4] == b"glTF" or b.startswith(b"{")   # glb magic or gltf json
```

🔁 Empty mesh → tessellation not run before export (`BRepMesh_IncrementalMesh`).

---

## Phase 7 — Frontend: upload → results → dual bars

🎯 Deliverables — `SearchBar.tsx` (upload + text + stage spinner), `ResultCard.tsx` (dual score
bars, feature chips, Near-dup/Weak-match badges), `api/client.ts`, `Search.tsx`. UI spec in
[`technicalImplementation.md` → Frontend](technicalImplementation.md).

✅ Gate — `frontend/tests/search.spec.ts` (Playwright, headless, seeded backend)
```
upload fixtures/bracket.step → at least one ResultCard renders
card shows a geo score bar and a text score bar
a card with geo≥0.95 shows the "Near-duplicate" badge
```

🔁 CORS error → `CORS_ORIGINS` / Vite proxy. Blank results → response shape mismatch with `client.ts`.

---

## Phase 8 — Side-by-side viewer + Costing prior + badges

🎯 Deliverables — `Viewer3D.tsx` (two-up glTF), `CostingPrior.tsx` (cost band from top-3, supplier
hint, DFM-from-history line, **Copy context for Costing**), duplicate banner. Details + stub logic
in [`technicalImplementation.md` → Business Extensions](technicalImplementation.md).

✅ Gate — Playwright
```
select top result → query|result viewers both mount
prior rail shows cost band "($min–$max, n=…)" and supplier hint
click "Copy context for Costing" → clipboard contains the part name + cost
```

🔁 Clipboard blocked in headless → use the Clipboard API mock / assert the payload builder output.

---

## Phase 9 — LLM adapter (flagged) + template fallback

🎯 Deliverables — `app/core/llm_adapter.py`: `LLM_PROVIDER ∈ {anthropic, gemini, none}`, one
`explain()` entry, SQLite cache by `(query_hash, result_id)`, template fallback. Default `none`
(demo cannot die on auth). See [`decisions.log`](decisions.log) and
[`questions.md`](questions.md) (Gemini endpoint is the likely provider).

✅ Gate — `backend/tests/unit/test_llm.py`
```python
def test_template_fallback_no_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER","none")
    txt = explain(qmeta, rmeta, {"geo":0.82,"text":0.41})
    assert "Geometry match 82%" in txt and len(txt) > 0
def test_cache_hit_no_second_call():   # second identical call served from SQLite
    ...
```

🔁 Fallback empty → template not reached when key missing. Re-billing → cache key wrong.

---

## Phase 10 — Full-product E2E + demo dress rehearsal

🎯 Deliverables
- `scripts/demo_smoke.py` (or Playwright spec) running the 5 demo steps from
  [`testing_demo.md` → Demo Script](testing_demo.md) headless against the full 30-part index.
- Full-library `eval_retrieval.py` run meeting [`../PRD.md` §6](../PRD.md) targets.
- 90-second backup recording.

✅ Gate
```bash
python scripts/index_library.py --step_dir data/demo_steps --metadata data/demo_metadata.json --output backend/app/data
python scripts/eval_retrieval.py            # targets met on full library
python scripts/demo_smoke.py                # all 5 steps pass
```

🔁 A demo step flakes → fix the data (metadata tokens, a bad STEP), not the test. Log any tuned
default in `decisions.log`.

---

## CI/CD rollout (fill as phases land)

`.github/workflows/ci.yml` — one workflow, jobs added when their phase is reached:

| Added at | Job | Runs |
|---|---|---|
| Phase 0 | `lint` | ruff + black --check + eslint |
| Phase 0 | `backend-unit` | `pytest backend/tests/unit` (grows each phase) |
| Phase 5 | `backend-api` | `pytest backend/tests/api` (TestClient, no network) |
| Phase 4 | `metrics` (nightly/opt-in) | install OCC+DGL, run integration + reduced `eval_retrieval.py`, upload `eval_report.json` artifact |
| Phase 7 | `frontend` | `npm ci && npm run build` + Playwright smoke on seeded index |

Rules: `main` must be green; heavy OCC/DGL job is nightly or `workflow_dispatch` (keeps PR CI fast);
cache the conda/pip env. Keep it minimal — this is a POC, CI proves the slices, it isn't the product.

---

## Progress checklist

- [ ] P0 skeleton + CI shell green
- [ ] P1 embedder gate
- [ ] P2 occ_stats + text gate
- [ ] P3 dual index + fusion gate
- [ ] P4 indexer + metrics gate (Recall@3 ≥ 0.75)
- [ ] P5 API gate
- [ ] P6 mesh/thumbnail gate
- [ ] P7 frontend results gate
- [ ] P8 viewer + costing prior gate
- [ ] P9 LLM + fallback gate
- [ ] P10 full E2E + demo rehearsal
