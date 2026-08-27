"""POST /api/search/cad (file, optional text) and POST /api/search/text."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_MB, TOP_K, WEAK_THRESHOLD
from app.core.index_singleton import get_index
from app.core.search_index import SearchHit, fuse

router = APIRouter()

_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Internal retrieval size per modality before fusion. Fusing over only the
# display k causes parts absent from one modality's top-k to default to a
# fusion score of 0, which looks like "0% similarity" rather than "wasn't
# in this shortlist". Pooling a larger candidate set (capped at index size)
# gives every candidate a real score from both modalities.
_FUSION_POOL = 50


class TextQuery(BaseModel):
    q: str
    k: int = TOP_K


def _hit_to_dict(hit: SearchHit, geo_available: bool = True) -> dict:
    part = hit.part
    badge = None
    if hit.is_duplicate:
        badge = "near-duplicate"
    elif hit.final_score < WEAK_THRESHOLD:
        badge = "weak-match"

    return {
        "id": part.id,
        "name": part.name,
        "material": part.material,
        "process": part.process,
        "cost": part.cost,
        "supplier": part.supplier,
        "notes": part.notes,
        "known_issues": part.known_issues,
        "ppap_notes": part.ppap_notes,
        "histogram": part.histogram,
        "occ_stats": part.occ_stats,
        "geo_score": round(hit.geo_score, 4) if geo_available else None,
        "text_score": round(hit.text_score, 4),
        "final_score": round(hit.final_score, 4),
        "badge": badge,
        "mesh_path": part.mesh_path,
        "thumb_path": part.thumb_path,
    }


@router.post("/search/cad")
async def search_cad(
    file: UploadFile = File(...),
    text: str = Form(default=""),
    k: int = Query(default=TOP_K, ge=1, le=20),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {ALLOWED_EXTENSIONS} files accepted")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit")

    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from app.core.graph_builder import step_to_dgl_graph
        from app.core.occ_stats import compute as occ_stats
        from app.core.uvnet_embedder import embed
        from app.occwl.io import load_shell

        try:
            graph = step_to_dgl_graph(tmp_path)
            geo_vec, histogram = embed(graph)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(422, str(e)) from e
        except Exception as e:
            raise HTTPException(
                422,
                "This STEP has freeform surfaces we can't convert to a B-rep graph. "
                "Try a prismatic solid (milled or turned part).",
            ) from e

        try:
            solids = load_shell(str(tmp_path))
            query_stats = occ_stats(solids[0].topods_shape()) if solids else {}
        except Exception:
            query_stats = {}

        idx = get_index()
        pool = min(idx.count(), max(k, _FUSION_POOL))
        geo_hits = idx.search_cad(geo_vec.numpy(), k=pool)

        from app.core.text_embedder import embed_text

        hist_tokens = " ".join(f"{count} {feat}" for feat, count in histogram.items() if count > 0)
        # User-typed text (if any) takes priority but both feed the same text
        # vector, so a combined CAD+text query narrows on both signals at once.
        user_typed = bool(text.strip())
        text_query = f"{text.strip()} {hist_tokens}".strip()
        text_vec = embed_text(text_query) if text_query else None

        # What actually produced the text signal — the UI/LLM must not claim
        # "geometry only" when the histogram alone drove a real text score.
        if user_typed and hist_tokens:
            text_source = "user_and_histogram"
        elif user_typed:
            text_source = "user"
        elif hist_tokens:
            text_source = "auto_histogram"
        else:
            text_source = "none"

        if text_vec is not None:
            text_hits = idx.search_text(text_vec, k=pool)
            geo_scores = {h.part.id: h.geo_score for h in geo_hits}
            text_scores = {h.part.id: h.text_score for h in text_hits}
            fused = fuse(geo_scores, text_scores)
            # build map from geo_hits first; backfill text_score from text_hits
            hit_map = {h.part.id: h for h in geo_hits}
            for th in text_hits:
                if th.part.id in hit_map:
                    hit_map[th.part.id].text_score = th.text_score
                else:
                    hit_map[th.part.id] = th
            results = []
            for pid, fscore in list(fused.items())[:k]:
                h = hit_map.get(pid)
                if h:
                    h.final_score = fscore
                    h.is_duplicate = h.geo_score >= 0.95
                    results.append(_hit_to_dict(h))
        else:
            for h in geo_hits:
                h.final_score = h.geo_score
            results = [_hit_to_dict(h) for h in geo_hits[:k]]

        latency_ms = round((time.perf_counter() - t0) * 1000)
        return JSONResponse(
            {
                "results": results,
                "query_histogram": histogram,
                "query_occ_stats": query_stats,
                "text_source": text_source,
                "latency_ms": latency_ms,
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/search/text")
def search_text(body: TextQuery):
    if not body.q.strip():
        raise HTTPException(400, "Query string is empty")

    from app.core.text_embedder import embed_text

    t0 = time.perf_counter()
    text_vec = embed_text(body.q)
    idx = get_index()
    hits = idx.search_text(text_vec, k=body.k)
    for h in hits:
        h.final_score = h.text_score
    latency_ms = round((time.perf_counter() - t0) * 1000)
    return {
        "results": [_hit_to_dict(h, geo_available=False) for h in hits],
        "text_source": "user",
        "latency_ms": latency_ms,
    }
