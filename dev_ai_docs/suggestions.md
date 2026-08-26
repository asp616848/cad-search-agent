# Suggestions — Validation + Cheap, High-Impact Improvements

Read this after `technicalImplementation.md`, `testing_demo.md`, `Business.md`, `repo_design.md`, and `decisions.log`. Goal: keep the POC small, make the demo undeniable, and close a few holes that would otherwise look like bugs in an interview.

Priority legend: **P0** = fix before you write much code · **P1** = do for the demo, still cheap · **P2** = nice if time leftover.

---

## 1. Validation of the current docs

### What is already strong

- **Product story is coherent.** Search-before-costing is a real gap, and the “compounding library / switching cost” argument in `Business.md` is the right one for Pre6.
- **Stack is interview-explainable.** FastAPI + pythonOCC/occwl + UV-Net + MiniLM + FAISS + SQLite is a local POC you can actually run on an M4 CPU.
- **Repo choices are correct for v1.** Lift graph building from `cad-feature-detection`, do not take Palmetto’s C++ engine, defer AAGNet. That matches `decisions.log`.
- **Demo script is good.** Near-duplicate, non-match, and “this feeds Costing” are the three moments that sell the product. Keep them.
- **Known issues are honest.** Multi-body STEP, cubes clustering, spline failures — call these out in the demo instead of hoping they don’t appear.

### Cross-doc inconsistencies to resolve (cheap)

| Issue | Where | Why it matters |
|---|---|---|
| **Weighted-sum vs concatenate is impossible as written** | `technicalImplementation.md` §B and `decisions.log` | You cannot do `0.7 * geo + 0.3 * text` *and* concatenate. Geo is 128–512-d, text is 384-d. Concatenate makes an 896-d (or similar) vector; weighted sum requires the same dimension. As written, the first person to implement this will pick one and the other docs will be wrong. |
| **Text-only search on a “384-dim slice” of a concat index** | `technicalImplementation.md` §C | If the FAISS index is concatenated geo+text, a text-only query that zeros the geo dims will match *geometry*, not text. Text search needs its **own** index (or score fusion). |
| **UV-Net embedding is not 512-d mean-pool** | `technicalImplementation.md` “Key Detail” | In this workspace, `UVNetGraphEncoder.forward` already returns `(node_emb, graph_emb)`. `graph_emb` is max-pooled across GNN layers and is `graph_emb_dim` (**128 by default**, not 512). Mean-pooling last-layer node features throws away the pooling the model was trained with. Use `graph_emb`, L2-normalize it. |
| **Business/ref promise drawings + PPAP files; v1 only indexes a notes string** | `Business.md`, `ref.md` vs implementation | Fine for a POC if you say so out loud. If you claim “multimodal knowledge base” and only search STEP + three metadata fields, the demo under-delivers. Either add a cheap document hook (below) or tighten the business copy to “CAD + structured manufacturing metadata, documents in v2.” |
| **AAGNet: ref.md says keep it; everything else defers it** | `ref.md` vs `repo_design.md` | Stick with defer. You already paid for face labels via UV-Net’s segmentation head (see §3). Do not add a second GNN pipeline for the POC. |
| **LLM choice is still open** | `questions.md` vs implementation | Claude vs Gemini vs “internal Claude” does not change architecture. Pick **one** env var (`LLM_PROVIDER`) and a 20-line adapter. Default to whatever key you already have so the demo cannot die on auth. |
| **Production note “embeddings pre-computed on upload, not on search query”** | `technicalImplementation.md` | Library embeddings: yes, precompute. **Query** embeddings: must be live. The sentence is easy to misread; split it. |

### Demo / testing gaps (not wrong, incomplete)

- Library mix of 30 parts is right; there is **no retrieval eval**. You will not know if UV-Net actually ranks the near-duplicate first until you are on stage.
- MFCAD filenames are feature-class dumps, not “L-Bracket Rev3.” `demo_metadata.json` is doing all the storytelling — it has to be written carefully or text search will look random.
- Palmetto `examples/test-models/` is documented as a source; that folder is often empty (README says drop files there). Prefer `cad-feature-detection` files + MFCAD + AAGNet examples as the real on-disk set.
- No plan for **what the UI shows if Anthropic/Gemini is down**. Explanations should degrade, search should not.

None of this requires a rewrite. It requires tightening the embedding contract and adding a few demo-grade features below.

---

## 2. Architecture — small changes, large payoff

### P0. Two indexes + score fusion (replace concat/weighted-sum)

Keep geometry and text in **separate FAISS indexes** (or two numpy matrices + brute-force L2, same thing at 30–100 parts).

```
cad_score  = cosine(q_geo,  library_geo)
text_score = cosine(q_text, library_text)   # 0 if user typed nothing
final      = 0.7 * cad_score + 0.3 * text_score   # now legal: both are scalars
```

- CAD-only upload: `final = cad_score`.
- Text-only: `final = text_score` (search the text index only).
- Both: fusion as above.
- Return **both scores** to the UI (see UX). This is the single best interview talking point after “B-rep embeddings, not pixels.”

Weights stay heuristic. Store them in config, not magic numbers in the search function.

### P0. Use UV-Net’s real graph embedding + keep the segmentation head

Do **not** invent a 512-d mean-pool. On each solid:

1. `node_emb, graph_emb = model.graph_encoder(...)` → **retrieval vector** (`graph_emb`, normalized).
2. Run the existing **segmentation head** (already in the checkpoint you are loading) → per-face labels → a compact **feature histogram** (`holes: 4, fillets: 6, pockets: 1, …`).

Histogram uses are all cheap:

- Show chips on result cards (“both have through-holes + fillets”).
- Concatenate a one-line summary into the MiniLM string so text search isn’t only fake notes.
- Optional: Jaccard overlap of label sets as a third score (`0.1` weight). Explains similarity without AAGNet.

This is the “feature understanding layer” `ref.md` wanted, without a second model.

### P1. Cheap geometric fingerprints (no extra ML)

At index time, pythonOCC already has the solid. Store:

- face / edge counts
- bounding-box aspect ratios
- volume and surface area (and `volume / bbox_volume` as a solidity ratio)

Use them as **tie-breakers** and as explanation fodder (“similar envelope, similar face count”). They also catch the cube problem: if UV-Net scores cluster, sort by bbox/volume among the top-k.

### P1. Index-time vs query-time, made explicit

| When | Work |
|---|---|
| Offline `index_library.py` | STEP → graph → UV-Net geo + histogram + OCC stats → MiniLM(metadata) → write FAISS + SQLite + glTF + thumbnail PNG |
| Upload query | Same geo path for the **query only**; never re-embed the library |
| Text query | MiniLM only; skip OCC/UV-Net |

Add a `GET /api/health` that reports: model loaded, index size, last index time. Saves 10 minutes of “why is search empty.”

### P1. LLM is optional, not on the critical path

```
search response = ranked hits + geo/text scores + feature overlap
if LLM_API_KEY: attach 2–3 sentence "why similar"
else: template from scores + overlapping tags + closest cost/supplier
```

Cache explanations by `(query_hash, result_id)` in SQLite. Repeat demo clicks should not re-bill the API.

Provider adapter: Anthropic **or** Gemini behind one function. Matches `questions.md` without a product change.

### P2. Assemblies without a new architecture

If STEP has multiple solids: embed **each body**, store `parent_id`. Query: embed all bodies, take **max** (or mean of top body-scores) against the library. Log the warning you already planned. One extra loop, much better than “first body only” if someone drops an assembly in the demo.

---

## 3. UX / design — what makes the demo feel like a product

### P1. Side-by-side 3D, not a single expanded card

Search is a comparison task. Layout:

- **Left (sticky):** query part viewer + filename + optional text.
- **Right:** ranked list; selected row opens **query | result** two-up glTF, aligned to the same camera if you can (even “fit both in view” is enough).

Engineers decide with their eyes. A grid of cards with a % is a gallery; two-up is a tool.

### P1. Show *why*, visually, without waiting on the LLM

On each result card:

- Dual bar: **Geometry 82% · Text 41%**
- Feature chips that overlap with the query (green) vs only on one side (muted)
- Badge: **Near-duplicate** if geo cosine ≥ 0.95; **Weak match** if top score &lt; 0.55 (your Step 5 script)

The LLM paragraph is a bonus *below* this. If the API is slow, the card still looks smart.

### P1. “Costing prior” panel (the business punch in one component)

When a result is selected, a right-hand rail:

- Historical cost (and **range from top-3**: min / median / max — 10 lines, huge narrative)
- Process, material, supplier
- `known_issues` / PPAP notes from metadata
- Button: **Copy context for Costing** → copies a JSON/plaintext blob:

```
Similar to "L-Bracket Rev3" (geo 0.91). Process: CNC milling. Quoted $420.
Supplier: Acme. Issue: thin wall at pocket. Use as prior for Costing.
```

You do not need to integrate Costing. Clipboard + a sentence in the demo is the integration story.

### P1. Library as the second half of stickiness

Already planned; add:

- Thumbnail generated at index time (render one glTF frame or a simple OCC screenshot) so the grid is instant.
- **Find similar** on a library part (reuses `/search/cad` with a stored embedding — no re-run of UV-Net if you save `geo_emb` on the row).
- Facet chips: material / process (SQLite `WHERE` on 30 rows; skip vector math).

### P2. Empty and error states that sound like an engineer, not a traceback

- Spline/occwl failure: “This STEP has geometry we can’t turn into a B-rep graph yet (often freeform surfaces). Try a solid-body mill/turn part.”
- Nothing similar: “No close geometry in this library. That’s expected for a first-of-kind part — start a fresh Costing run.” (matches `Business.md` fork)
- Multi-body: banner “Embedded N bodies; ranking uses the closest body.”

### P2. Upload UX

- Accept `.step` / `.stp` / `.STEP`; reject with a one-liner otherwise.
- 25 MB cap + spinner with **stage text**: “Reading B-rep → UV-Net → search” (hides 1–3s CPU).
- Keep the last query on screen when results arrive (people demo with their back to the laptop).

---

## 4. Implementation — cheap hardening

### P0. One golden retrieval script

`scripts/eval_retrieval.py` with ~8 labeled queries:

| Query | Must appear in top-3 |
|---|---|
| Near-duplicate STEP | its pair, score ≥ 0.95 |
| Bracket STEP | other brackets, not shafts |
| Text: “M6 aluminum bracket” | the parts whose notes mention M6 + Al |
| Shaft STEP | shafts |
| Cube / block | do not claim a hero score; assert you show a weak-match state |

Print Recall@3 and the ranked names. Run it before the interview. This is the difference between hoping UV-Net works and *knowing*.

### P1. Build the near-duplicate pair yourself

Do not hunt MFCAD for “slight variation.” Copy one known-good STEP, open in FreeCAD (or ask someone with CAD for 5 minutes), add a fillet or hole, export `bracket_001_rev2.step`. That pair *is* the duplicate-detection slide.

### P1. Metadata that text search can actually hit

`demo_metadata.json` should be written as **search documents**, not labels:

- Put tokens an engineer would type: `M6`, `counterbore`, `6061`, `anodize`, `PPAP`, `thin wall`.
- Add `known_issues` and `ppap_notes` fields even if fake — they power the Costing-prior rail and MiniLM.
- Include 2–3 parts that share a supplier so “supplier recommendation” is a live sentence, not a roadmap slide.

### P1. Index script prints a library card

After indexing: count by process/material, min/max cost, list of filenames that failed. Failed STEP files should not silently vanish from the demo set.

### P1. Frontend: similarity is cosine, not a mysterious %

If you display `%`, document `percent = 50 * (cosine + 1)` or better **just show 0–1 cosine** plus the dual bars. Fake 97% from uncalibrated L2 is an easy way to lose trust with a technical audience.

### P2. Upload safety

Max size, timeout on UV-Net, temp file cleanup, do not persist query STEP unless they click “add to library.”

### P2. `POST /api/library/index` from the UI

One “Add this query to the library” after a search. Completes the knowledge-loop story in 30 seconds of live demo: search → add → search again → it is now hit #1. That is stickiness, on camera.

---

## 5. Testing & demo — extras that cost almost no time

- **Smoke order:** (1) isolated embed script in `testing_demo.md`, (2) `eval_retrieval.py`, (3) UI path. Do not open Vite until (1) and (2) pass.
- **Timebox talking points:** log `graph_ms`, `uvnet_ms`, `faiss_ms` on the search response. “CPU, 400ms, no GPU” is a better answer than “it’s fast.”
- **Backup recording:** 90-second screen recording of the five demo steps. Wifi / API / STEP path issues happen.
- **One “ugly” part in the library** that you *expect* to fail (spline or multi-body) so you can show the error state on purpose.
- Confirm MFCAD clone path; have a **local zip** of 30 STEP files in `eng-search/data/demo_steps/` so the demo is not git-clone-from-GitHub on interview morning.

---

## 6. Business / product add-ons that fit this POC

These are not new products. They are **one UI element + metadata** that make the Pre6 story true.

### P1. Cost band from neighbors (extension #2, stubbed)

From top-k with geo score ≥ 0.7, show “Historical quotes: $380–$510 (n=4).” You already store `cost`. This is cost prediction without a model. Say it is a **prior**, not a quote.

### P1. Duplicate gate (extension #1, stubbed)

If max geo cosine ≥ 0.95: banner **“This looks already quoted.”** Link the library part. That is the “prevent redundant work” bullet, live.

### P1. Supplier hint (extension #3, stubbed)

“2 of 3 closest parts went to Acme Machining.” Count suppliers in top-k. No recommender system.

### P1. Historical DFM line (extension #4, stubbed)

If `known_issues` on a neighbor shares a feature tag with the query histogram (e.g. both have `thin_wall` / pocket): highlight that issue in the prior rail. Fake the tags in metadata; the **mechanism** is real.

### P1. Tighten the journey copy

Keep `Business.md` as-is for vision, but add one sentence for v1 scope:

> v1 retrieves similar solids and their manufacturing metadata and **exports a Costing prior**. Drawings, PPAP PDFs, and automatic Costing API injection are the same loop, one integration later.

That keeps the Adobe-stickiness metaphor honest.

### P2. Search log table

`searches(id, timestamp, mode, top_ids, clicked_id)`. No UI required. Story: “production learns the 0.7/0.3 weights from clicks.” You can even click during the demo and show the row in SQLite.

### P2. Drawings without a VLM

If you want a multimodal checkmark without VLM-CAD: accept a PDF, `pypdf` extract text, MiniLM that text into the **text index**. Title-block notes often contain part names and materials. Skip image embeddings for v1.

---

## 7. What not to do for this POC

- AAGNet / gAAG pipeline
- Palmetto C++ engine
- Qdrant / Postgres / GPU workers
- Fine-tuning UV-Net
- OpenAI large embeddings
- Full Costing/Machinist API integration
- Auth / multi-tenant (mention in `decisions.log` only)
- Pixel/VLM search as a third index

All of these are already correctly deferred. Do not let `ref.md` pull them back in.

---

## 8. Suggested implementation order (still a short POC)

1. Graph lift + **true `graph_emb`** + health check + eval script on 10 files.  
2. Dual FAISS + SQLite + metadata + histograms + OCC stats.  
3. glTF + two-up viewer + dual score bars + badges.  
4. Costing-prior rail + copy button + duplicate banner + cost band.  
5. LLM explanation behind a flag, with template fallback.  
6. “Add query to library” + thumbnails + text-only path.  
7. Write the 30-row `demo_metadata.json` last, against the eval script, not first.

---

## 9. One-line north star for the interview

> We turn every past STEP + quote into a vector you can search *before* Costing, with geometry and manufacturing language as separate signals, and we hand Costing a grounded prior — not a blank slate.

If a feature does not make that sentence truer, skip it for v1.
