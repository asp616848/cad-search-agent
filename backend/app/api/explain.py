"""POST /api/explain — why-similar paragraph; POST /api/explain/ask — follow-up Q&A."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import DATA_DIR
from app.core.index_singleton import get_index
from app.core.llm_adapter import _describe_query, _facts_line, explain

router = APIRouter()


class ExplainRequest(BaseModel):
    result_id: int
    geo_score: float | None = None
    text_score: float = 0.0
    # "none" | "auto_histogram" | "user" | "user_and_histogram" — ground
    # truth from the search response, not a frontend guess.
    text_source: str = "none"
    query_text: str = ""
    query_histogram: dict[str, int] = {}
    query_occ_stats: dict = {}


class AskRequest(BaseModel):
    result_id: int
    question: str
    geo_score: float | None = None
    text_score: float = 0.0
    text_source: str = "none"
    query_text: str = ""
    query_histogram: dict[str, int] = {}
    query_occ_stats: dict = {}


def _result_meta(part) -> dict:
    return {
        "name": part.name,
        "material": part.material,
        "process": part.process,
        "notes": part.notes,
        "known_issues": part.known_issues,
        "histogram": part.histogram,
        "occ_stats": part.occ_stats,
    }


def _thumbnail_path(result_id: int):
    path = DATA_DIR / "thumbnails" / f"{result_id}.png"
    return path if path.exists() else None


@router.post("/explain")
def post_explain(body: ExplainRequest):
    idx = get_index()
    part = idx._fetch_part(body.result_id)
    if part is None:
        raise HTTPException(404, f"Part {body.result_id} not found")

    text = explain(
        query_meta={
            "text_source": body.text_source,
            "text": body.query_text,
            "histogram": body.query_histogram,
            "occ_stats": body.query_occ_stats,
        },
        result_meta=_result_meta(part),
        scores={"geo": body.geo_score, "text": body.text_score},
        result_image_path=_thumbnail_path(body.result_id),
    )
    return {"explanation": text}


@router.post("/explain/ask")
def post_ask(body: AskRequest):
    if not body.question.strip():
        raise HTTPException(400, "Question is empty")

    idx = get_index()
    part = idx._fetch_part(body.result_id)
    if part is None:
        raise HTTPException(404, f"Part {body.result_id} not found")

    from app.config import LLM_API_KEY, LLM_PROVIDER

    if LLM_PROVIDER == "none" or not LLM_API_KEY:
        return {
            "answer": (
                "LLM provider is not configured (LLM_PROVIDER=none). "
                "Set LLM_PROVIDER and LLM_API_KEY to enable follow-up questions."
            )
        }

    meta = _result_meta(part)
    query_meta = {
        "text_source": body.text_source,
        "text": body.query_text,
        "histogram": body.query_histogram,
        "occ_stats": body.query_occ_stats,
    }
    geo_present = body.geo_score is not None
    geo_sentence = (
        "Geometry was not evaluated for this query (it was a text-only search)."
        if not geo_present
        else f"Geometry similarity: {round(body.geo_score * 100)}%."
    )
    result_facts = _facts_line(meta["histogram"], meta["occ_stats"])
    prompt = (
        f"{_describe_query(query_meta, geo_present)} "
        f"The candidate result is '{meta['name']}' ({meta['material']}, {meta['process']}) "
        f"with {result_facts}. "
        f"{geo_sentence} Text/metadata similarity: {round(body.text_score * 100)}%. "
        f"Notes: {meta['notes']}. Known issues: {meta['known_issues']}. "
        f'The user asks: "{body.question}"\n'
        f"Answer concisely (2-4 sentences), grounded only in the info above (including the "
        f"attached image if present). Do not claim a geometry comparison happened if it did not."
    )

    image_path = _thumbnail_path(body.result_id)
    try:
        if LLM_PROVIDER == "anthropic":
            from app.core.llm_adapter import _call_anthropic

            answer = _call_anthropic(prompt, image_path)
        elif LLM_PROVIDER == "gemini":
            from app.core.llm_adapter import _call_gemini

            answer = _call_gemini(prompt, image_path)
        else:
            answer = "Unknown LLM_PROVIDER."
    except Exception as e:
        answer = f"Could not reach the LLM provider: {e}"

    return {"answer": answer}
