"""Phase 3 gate — dual FAISS index, fusion, SQLite."""

from pathlib import Path

import numpy as np

from app.core.search_index import SearchIndex, fuse

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _make_index(tmp_path: Path) -> tuple[SearchIndex, list[int]]:
    """Build a 3-part in-memory index from fixtures. Returns (index, [idA, idB, idC])."""
    from app.core.graph_builder import step_to_dgl_graph
    from app.core.text_embedder import embed_text
    from app.core.uvnet_embedder import embed

    idx = SearchIndex(db_path=tmp_path / "parts.db")
    ids = []
    for stem, meta in [
        ("partA", "aluminum bracket machined"),
        ("partB", "steel shaft turned"),
        ("partC", "titanium plate milled"),
    ]:
        graph = step_to_dgl_graph(FIXTURE_DIR / f"{stem}.step")
        geo_vec, hist = embed(graph)
        text_vec = embed_text(meta)
        pid = idx.add(
            name=stem,
            geo_vec=geo_vec.numpy(),
            text_vec=text_vec,
            material=meta,
            histogram=hist,
        )
        ids.append(pid)
    return idx, ids


def test_part_finds_itself(tmp_path: Path) -> None:
    from app.core.graph_builder import step_to_dgl_graph
    from app.core.uvnet_embedder import embed

    idx, (id_a, id_b, id_c) = _make_index(tmp_path)
    graph = step_to_dgl_graph(FIXTURE_DIR / "partA.step")
    geo_vec, _ = embed(graph)
    hits = idx.search_cad(geo_vec.numpy(), k=3)
    assert len(hits) >= 1
    assert hits[0].part.id == id_a
    assert hits[0].geo_score > 0.99


def test_text_search_returns_results(tmp_path: Path) -> None:
    from app.core.text_embedder import embed_text

    idx, _ = _make_index(tmp_path)
    text_vec = embed_text("aluminum bracket")
    hits = idx.search_text(text_vec, k=3)
    assert len(hits) >= 1
    assert hits[0].text_score > 0.0


def test_fusion_math() -> None:
    # weights 0.7/0.3 (config defaults)
    result = fuse({1: 0.8, 2: 0.2}, {1: 0.0, 2: 1.0})
    assert abs(result[1] - 0.56) < 1e-5
    assert abs(result[2] - 0.44) < 1e-5
    # result[1] should rank above result[2]
    ranked = list(result.keys())
    assert ranked[0] == 1


def test_fusion_custom_weights() -> None:
    result = fuse({1: 1.0, 2: 0.0}, {1: 0.0, 2: 1.0}, geo_weight=0.5, text_weight=0.5)
    assert abs(result[1] - 0.5) < 1e-5
    assert abs(result[2] - 0.5) < 1e-5


def test_index_count(tmp_path: Path) -> None:
    idx, _ = _make_index(tmp_path)
    assert idx.count() == 3


def test_empty_index_returns_no_hits(tmp_path: Path) -> None:
    idx = SearchIndex(db_path=tmp_path / "empty.db")
    vec = np.random.randn(128).astype(np.float32)
    assert idx.search_cad(vec, k=5) == []
    assert idx.search_text(np.random.randn(384).astype(np.float32), k=5) == []
