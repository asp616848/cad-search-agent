# cad-search-agent — Product & Engineering Spec

Engineering knowledge search for Pre6: turn every past STEP + quote into a vector you can
search **before** Costing, with geometry and manufacturing language as separate signals, and
hand Costing a grounded prior instead of a blank slate.

This file is the single entry point. Detail already written elsewhere is **referenced, not
repeated** — follow the links.

---

## 1. Source-of-truth documents

| Topic | Document |
|---|---|
| Business context, Pre6 journey, v1 scope, extensions | [`dev_ai_docs/Business.md`](dev_ai_docs/Business.md) |
| Which repos we use/defer and why | [`dev_ai_docs/repo_design.md`](dev_ai_docs/repo_design.md) |
| Stack, data flow, UV-Net usage, API, config, UI spec | [`dev_ai_docs/technicalImplementation.md`](dev_ai_docs/technicalImplementation.md) |
| Every non-obvious decision + tradeoff | [`dev_ai_docs/decisions.log`](dev_ai_docs/decisions.log) |
| Datasets, demo library, demo script, smoke order | [`dev_ai_docs/testing_demo.md`](dev_ai_docs/testing_demo.md) |
| **Phased build plan with per-phase test gates + self-improving loop** | [`dev_ai_docs/implementation_plan.md`](dev_ai_docs/implementation_plan.md) |
| Open product questions | [`dev_ai_docs/questions.md`](dev_ai_docs/questions.md) |
| Original brainstorm | [`dev_ai_docs/ref.md`](dev_ai_docs/ref.md) |

If any statement here conflicts with those files, **those files win** and this file gets fixed.

---

## 2. What we are building (one screen)

Upload a STEP file (and/or type a description) → get ranked similar past parts, each with:
geometry + text match scores, overlapping machining features, historical cost/supplier/issues,
and a one-click "Costing prior" export. See UI spec in
[`technicalImplementation.md` → Frontend](dev_ai_docs/technicalImplementation.md).

**Non-goals for v1** (deferred, see `decisions.log`): AAGNet, Palmetto C++ engine, Qdrant/Postgres,
GPU, UV-Net fine-tuning, VLM image search, auth/multi-tenant, Costing API integration.

---

## 3. Architecture at a glance

```
                         ┌──────────────────────────────────────────┐
  STEP upload ──────────▶│ backend/app/core                         │
  or text query          │  graph_builder → uvnet_embedder (geo+hist)│
                         │  occ_stats · text_embedder                │
                         │  search_index (2× FAISS) · llm_adapter    │
                         └───────────────┬──────────────────────────┘
                                         │ FastAPI (app/api/*)
                                         ▼
   data/  ── geo_index.faiss · text_index.faiss · parts.db · meshes/ · thumbnails/
                                         ▲
  scripts/index_library.py ──────────────┘   (offline, builds the above)
  scripts/eval_retrieval.py ── metrics report (Recall@k, MRR) → tests/ gate

  frontend/  React + Vite + Three.js  ──HTTP──▶ FastAPI
```

Full file tree: [`repo_design.md` → Our Repo Structure](dev_ai_docs/repo_design.md).
Data flow (index-time vs query-time): [`technicalImplementation.md`](dev_ai_docs/technicalImplementation.md).

---

## 4. Build philosophy — thin vertical slices, gated

We do **not** build all of the backend, then all of the frontend, then debug. Each phase is a
thin slice that runs end-to-end and has an **explicit test gate** that must pass before the next
phase starts. A phase is "done" only when its gate is green.

The ordered phases, their deliverables, and the exact command that proves each one works live in
[`dev_ai_docs/implementation_plan.md`](dev_ai_docs/implementation_plan.md). Summary of the ladder:

| Phase | Slice | Proven by |
|---|---|---|
| 0 | Repo skeleton, config, CI shell | `pytest` collects, lint passes, CI green on empty |
| 1 | STEP → graph → `geo_vec` + histogram | `tests/test_embedder.py` on 3 on-disk STEPs |
| 2 | OCC stats + text embedder | unit tests assert shapes/ranges |
| 3 | Dual FAISS index + fusion + SQLite | `tests/test_search.py` known-part-finds-itself |
| 4 | `index_library.py` + `eval_retrieval.py` metrics | Recall@3 ≥ target on labeled set |
| 5 | FastAPI endpoints (`/health`, `/search/*`) | `tests/test_api.py` via TestClient |
| 6 | glTF + thumbnails served | endpoint returns valid glb bytes |
| 7 | Frontend: upload → results → dual bars | Playwright smoke: upload fixture → card renders |
| 8 | Side-by-side viewer + Costing prior + badges | Playwright: select result → prior rail + copy |
| 9 | LLM adapter (flag) + template fallback | test both paths; fallback needs no key |
| 10 | Full-product E2E + demo dress rehearsal | scripted 5-step demo passes headless |

---

## 5. Testing strategy (what lives where)

Testing is a **first-class deliverable**, not an afterthought. Layers:

- **Unit** (`backend/tests/unit/`) — pure functions: embedder shapes, score fusion math, OCC
  stat ranges, template explanation formatting. Fast, no model load where avoidable.
- **Integration** (`backend/tests/integration/`) — real STEP → real pipeline → real FAISS, using a
  tiny fixture set of 3–5 STEP files committed under `backend/tests/fixtures/`.
- **Retrieval metrics** (`scripts/eval_retrieval.py` + `data/eval_labels.json`) — the ML quality
  gate. Reports **Recall@1/3/5, MRR, and per-query rank** to `data/eval_report.json` and stdout.
  This is the number that tells us the product actually works, not just that code runs.
- **API** (`backend/tests/api/`) — FastAPI `TestClient`, no network.
- **Frontend E2E** (`frontend/tests/`) — Playwright, headless, against a seeded test index.
- **Full-product smoke** — one script that runs the 5 demo steps end to end
  ([demo script in `testing_demo.md`](dev_ai_docs/testing_demo.md)).

Fixtures and dataset sourcing: [`testing_demo.md`](dev_ai_docs/testing_demo.md).

---

## 6. Metrics we report

`eval_retrieval.py` is the scoreboard. It must print and persist:

| Metric | Meaning | v1 target |
|---|---|---|
| Recall@3 | labeled correct part in top-3 | ≥ 0.75 across the labeled set |
| MRR | mean reciprocal rank of the correct part | ≥ 0.70 |
| Near-dup rank | the built duplicate pair ranks #1 for each other | rank == 1, geo ≥ 0.90 |
| Weak-match precision | first-of-kind query returns top score < 0.55 | holds |
| Latency | `graph_ms`, `uvnet_ms`, `faiss_ms` (p50) | uvnet < 1500ms CPU |

Targets are tunable in `data/eval_labels.json` header. Missing a target is a **signal to loop**
(see §7), not a reason to ship anyway.

---

## 7. The self-improving loop

Each phase runs this loop until its gate is green, then advances:

```
implement slice
      │
      ▼
run the phase gate command  ──pass──▶ commit, advance to next phase
      │ fail
      ▼
inspect the metric/failure (eval_report.json, pytest output, Playwright trace)
      │
      ▼
form ONE hypothesis (weights? threshold? histogram enrichment? bad fixture?)
      │
      ▼
change ONE knob (config value, prompt, fusion weight, fixture) ──▶ re-run gate
```

Knobs are deliberately in `config.py` / `eval_labels.json` so tuning never touches search logic.
Every loop that changes a default gets a one-line entry appended to
[`decisions.log`](dev_ai_docs/decisions.log) so the "why" is never lost.

---

## 8. CI/CD

A single GitHub Actions workflow (`.github/workflows/ci.yml`) — kept minimal, added in Phase 0
as a shell and filled as phases land. Jobs:

1. **lint** — `ruff` + `black --check` (backend), `eslint` (frontend).
2. **backend-tests** — `pytest backend/tests/unit backend/tests/api` (unit + API; skip the heavy
   OCC/UV-Net integration job unless STEP fixtures + deps are cached).
3. **integration** (opt-in / nightly) — installs pythonOCC + DGL, runs integration + a reduced
   `eval_retrieval.py` on the committed fixture set, uploads `eval_report.json` as an artifact.
4. **frontend-build** — `npm ci && npm run build` + Playwright smoke against a seeded index.

CI must be green before merge to `main`. Detail and rollout-by-phase in
[`implementation_plan.md` → CI/CD rollout](dev_ai_docs/implementation_plan.md).

---

## 9. Definition of done (product)

- [ ] All 10 phase gates green
- [ ] `eval_retrieval.py` meets §6 targets on the 30-part library
- [ ] 5-step demo ([`testing_demo.md`](dev_ai_docs/testing_demo.md)) runs clean, plus a backup recording
- [ ] CI green on `main`
- [ ] `decisions.log` reflects every tuned default
