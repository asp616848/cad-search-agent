"""LLM adapter — explain() with SQLite cache and template fallback.

LLM_PROVIDER env var: anthropic | gemini | none (default)
Demo is safe to run with LLM_PROVIDER=none — template always returns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from app.config import DATA_DIR, LLM_API_KEY, LLM_PROVIDER

_CACHE_DB = DATA_DIR / "llm_cache.db"
_CREATE = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    explanation TEXT NOT NULL
);
"""


def _cache_db() -> sqlite3.Connection:
    _CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_DB), check_same_thread=False)
    conn.execute(_CREATE)
    conn.commit()
    return conn


_conn: sqlite3.Connection | None = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = _cache_db()
    return _conn


def _cache_key(query_meta: dict, result_meta: dict, scores: dict) -> str:
    payload = json.dumps({"q": query_meta, "r": result_meta, "s": scores}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _facts_line(histogram: dict, occ_stats: dict) -> str:
    """Render the structured geometry facts we actually have (feature
    histogram + OCC stats) as a short text blurb, so the LLM can reason
    about concrete feature counts/dimensions instead of only percentages."""
    parts = []
    if histogram:
        feat_str = ", ".join(f"{v}x {k.replace('_', ' ')}" for k, v in histogram.items() if v)
        if feat_str:
            parts.append(f"features: {feat_str}")
    if occ_stats:
        bits = []
        if occ_stats.get("face_count"):
            bits.append(f"{occ_stats['face_count']} faces")
        if occ_stats.get("volume"):
            bits.append(f"volume {occ_stats['volume']:.0f}mm3")
        bx, by, bz = occ_stats.get("bbox_x"), occ_stats.get("bbox_y"), occ_stats.get("bbox_z")
        if bx and by and bz:
            bits.append(f"bbox {bx:.0f}x{by:.0f}x{bz:.0f}mm")
        if occ_stats.get("solidity"):
            bits.append(f"solidity {occ_stats['solidity']:.2f}")
        if bits:
            parts.append("stats: " + ", ".join(bits))
    return "; ".join(parts) if parts else "no structured geometry facts available"


def _template(query_meta: dict, result_meta: dict, scores: dict) -> str:
    geo = scores.get("geo")
    text_pct = round(scores.get("text", 0) * 100)
    name = result_meta.get("name", "this part")
    material = result_meta.get("material", "")
    process = result_meta.get("process", "")
    mat_proc = f"{material} {process}".strip() or "similar process"

    text_sentence = (
        f"Text match {text_pct}% — {name} is a {mat_proc} part with overlapping metadata tokens."
    )
    if geo is None:
        return (
            f"This was a text-only search — no CAD file was uploaded, so no geometry "
            f"comparison was made. {text_sentence}"
        )
    geo_pct = round(geo * 100)
    return (
        f"Geometry match {geo_pct}% — the B-rep face-adjacency graph shares similar "
        f"machined feature topology. {text_sentence}"
    )


def _describe_query(query_meta: dict) -> str:
    """Describe what the user actually searched with, so the LLM never
    invents a CAD comparison that didn't happen (or vice versa)."""
    mode = query_meta.get("mode", "cad")
    text_query = query_meta.get("text", "")
    facts = _facts_line(query_meta.get("histogram", {}), query_meta.get("occ_stats", {}))

    if mode == "text":
        return (
            f'The user ran a TEXT-ONLY search for "{text_query}" — no STEP/CAD file was '
            f"uploaded, so no geometry comparison was performed at all."
        )
    if mode == "cad_text":
        return (
            f'The user uploaded a STEP/CAD file (query {facts}) and also typed "{text_query}" '
            f"as narrowing text — both geometry and text were compared."
        )
    return f"The user uploaded a STEP/CAD file (query {facts}) and searched by geometry only."


def _build_explain_prompt(query_meta: dict, result_meta: dict, scores: dict) -> str:
    geo = scores.get("geo")
    geo_sentence = (
        "Geometry was not evaluated for this query (see above)."
        if geo is None
        else f"Geometry similarity: {round(geo * 100)}%."
    )
    result_facts = _facts_line(result_meta.get("histogram", {}), result_meta.get("occ_stats", {}))
    return (
        f"{_describe_query(query_meta)} "
        f"The candidate result is '{result_meta.get('name')}' "
        f"({result_meta.get('material', '')}, {result_meta.get('process', '')}) "
        f"with {result_facts}. "
        f"{geo_sentence} Text/metadata similarity: {round(scores.get('text', 0) * 100)}%. "
        f"Using the concrete feature counts and dimensions above (not just the percentages), "
        f"explain in 2-3 sentences why this result surfaced and how its actual geometry "
        f"compares to the query's. Be specific about matching or differing feature types/counts "
        f"when both sides have them. Do not claim a geometry comparison happened if it did not. "
        f"Be concise and engineering-specific."
    )


def _call_anthropic(prompt: str, image_path: Path | None = None) -> str:
    import base64

    import anthropic  # type: ignore[import]

    client = anthropic.Anthropic(api_key=LLM_API_KEY)
    content: list = []
    if image_path and image_path.exists():
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(image_path.read_bytes()).decode(),
                },
            }
        )
    content.append({"type": "text", "text": prompt})

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=160,
        messages=[{"role": "user", "content": content}],
    )
    return msg.content[0].text.strip()


def _call_gemini(prompt: str, image_path: Path | None = None) -> str:
    import google.generativeai as genai  # type: ignore[import]

    genai.configure(api_key=LLM_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    parts: list = [prompt]
    if image_path and image_path.exists():
        from PIL import Image

        parts.append(Image.open(image_path))

    response = model.generate_content(parts)
    return response.text.strip()


def explain(
    query_meta: dict,
    result_meta: dict,
    scores: dict,
    result_image_path: Path | None = None,
) -> str:
    """Return a 2-3 sentence explanation of why the result matches the query.

    query_meta / result_meta may carry "histogram" and "occ_stats" keys —
    when present, these concrete feature counts and dimensions are folded
    into the prompt so the LLM reasons about real geometry, not just the
    similarity percentages. If result_image_path points at an existing
    thumbnail PNG, it's attached as a vision input (gemini/anthropic only).

    Falls back to template if LLM_PROVIDER is 'none' or call fails.
    Result is cached in SQLite keyed by (query_meta, result_meta, scores).
    """
    key = _cache_key(query_meta, result_meta, scores)
    conn = _get_conn()

    row = conn.execute("SELECT explanation FROM llm_cache WHERE cache_key=?", (key,)).fetchone()
    if row:
        return row[0]

    text = _template(query_meta, result_meta, scores)  # default

    if LLM_PROVIDER != "none" and LLM_API_KEY:
        prompt = _build_explain_prompt(query_meta, result_meta, scores)
        try:
            if LLM_PROVIDER == "anthropic":
                text = _call_anthropic(prompt, result_image_path)
            elif LLM_PROVIDER == "gemini":
                text = _call_gemini(prompt, result_image_path)
        except Exception:
            pass  # fall through to template already set

    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (cache_key, explanation) VALUES (?,?)",
        (key, text),
    )
    conn.commit()
    return text
