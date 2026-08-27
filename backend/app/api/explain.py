"""POST /api/explain — why-similar paragraph; POST /api/explain/ask — follow-up Q&A."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.index_singleton import get_index
from app.core.llm_adapter import explain

router = APIRouter()


class ExplainRequest(BaseModel):
    result_id: int
    geo_score: float = 0.0
    text_score: float = 0.0
    query_name: str = "uploaded part"


class AskRequest(BaseModel):
    result_id: int
    question: str
    geo_score: float = 0.0
    text_score: float = 0.0


def _result_meta(part) -> dict:
    return {
        "name": part.name,
        "material": part.material,
        "process": part.process,
        "notes": part.notes,
        "known_issues": part.known_issues,
    }


@router.post("/explain")
def post_explain(body: ExplainRequest):
    idx = get_index()
    part = idx._fetch_part(body.result_id)
    if part is None:
        raise HTTPException(404, f"Part {body.result_id} not found")

    text = explain(
        query_meta={"name": body.query_name},
        result_meta=_result_meta(part),
        scores={"geo": body.geo_score, "text": body.text_score},
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
    prompt = (
        f"A user is comparing a query CAD part to a candidate result part "
        f"'{meta['name']}' ({meta['material']}, {meta['process']}). "
        f"Geometry similarity: {round(body.geo_score * 100)}%. "
        f"Text similarity: {round(body.text_score * 100)}%. "
        f"Notes: {meta['notes']}. Known issues: {meta['known_issues']}. "
        f'The user asks: "{body.question}"\n'
        f"Answer concisely (2-4 sentences), grounded only in the info above."
    )

    try:
        if LLM_PROVIDER == "anthropic":
            from app.core.llm_adapter import _call_anthropic

            answer = _call_anthropic(prompt)
        elif LLM_PROVIDER == "gemini":
            from app.core.llm_adapter import _call_gemini

            answer = _call_gemini(prompt)
        else:
            answer = "Unknown LLM_PROVIDER."
    except Exception as e:
        answer = f"Could not reach the LLM provider: {e}"

    return {"answer": answer}
