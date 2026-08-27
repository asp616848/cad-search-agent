"""MiniLM-L6-v2 text embedder — 384-dim L2-normalized vectors."""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a metadata string to a 384-dim L2-normalized float32 vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


def histogram_to_text(histogram: dict[str, int]) -> str:
    """Single canonical string form for a feature histogram, used by every
    call site (query-time auto-text, index-time metadata, live library adds).

    Two different phrasings of the same histogram (counts vs no counts,
    underscores vs spaces) embed to measurably different vectors — even an
    exact self-search wouldn't hit ~100% text similarity if the query and
    the indexed document disagreed on formatting. Keeping one function
    means that can't drift apart again.
    """
    return " ".join(
        f"{count}x {feat.replace('_', ' ')}" for feat, count in histogram.items() if count > 0
    )
