"""LLM adapter — explain() with SQLite cache and template fallback.

LLM_PROVIDER env var: anthropic | gemini | none (default)
Demo is safe to run with LLM_PROVIDER=none — template always returns.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

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


def _template(query_meta: dict, result_meta: dict, scores: dict) -> str:
    geo_pct = round(scores.get("geo", 0) * 100)
    text_pct = round(scores.get("text", 0) * 100)
    name = result_meta.get("name", "this part")
    material = result_meta.get("material", "")
    process = result_meta.get("process", "")
    mat_proc = f"{material} {process}".strip() or "similar process"
    return (
        f"Geometry match {geo_pct}% — the B-rep face-adjacency graph shares similar "
        f"machined feature topology. Text match {text_pct}% — {name} is a {mat_proc} "
        f"part with overlapping metadata tokens."
    )


def _call_anthropic(prompt: str) -> str:
    import anthropic  # type: ignore[import]

    client = anthropic.Anthropic(api_key=LLM_API_KEY)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _call_gemini(prompt: str) -> str:
    import google.generativeai as genai  # type: ignore[import]

    genai.configure(api_key=LLM_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text.strip()


def explain(
    query_meta: dict,
    result_meta: dict,
    scores: dict,
) -> str:
    """Return a 2-3 sentence explanation of why the result matches the query.

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
        prompt = (
            f"In 2-3 sentences, explain why a CAD part '{result_meta.get('name')}' "
            f"({result_meta.get('material', '')}, {result_meta.get('process', '')}) "
            f"scored {round(scores.get('geo', 0)*100)}% geometry similarity and "
            f"{round(scores.get('text', 0)*100)}% text similarity to the query part. "
            f"Be concise and engineering-specific."
        )
        try:
            if LLM_PROVIDER == "anthropic":
                text = _call_anthropic(prompt)
            elif LLM_PROVIDER == "gemini":
                text = _call_gemini(prompt)
        except Exception:
            pass  # fall through to template already set

    conn.execute(
        "INSERT OR REPLACE INTO llm_cache (cache_key, explanation) VALUES (?,?)",
        (key, text),
    )
    conn.commit()
    return text
