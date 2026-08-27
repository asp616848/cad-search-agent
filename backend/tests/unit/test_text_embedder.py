"""Unit tests for text_embedder.embed_text()."""

import numpy as np


def test_text_vec_shape() -> None:
    from app.core.text_embedder import embed_text

    v = embed_text("M6 aluminum bracket")
    assert v.shape == (384,)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_text_vec_dtype() -> None:
    from app.core.text_embedder import embed_text

    v = embed_text("stainless steel shaft")
    assert v.dtype == np.float32


def test_different_texts_differ() -> None:
    from app.core.text_embedder import embed_text

    v1 = embed_text("aluminum bracket with holes")
    v2 = embed_text("titanium turbine blade")
    sim = float(np.dot(v1, v2))
    assert sim < 0.999
