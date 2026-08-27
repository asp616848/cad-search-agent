"""Phase 5 gate — FastAPI endpoints via TestClient with a seeded fixture index."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(scope="module")
def seeded_client(tmp_path_factory):
    """Build a small in-memory index, patch the singleton, return TestClient."""
    tmp = tmp_path_factory.mktemp("api_data")

    # Build index from fixtures
    from app.core.graph_builder import step_to_dgl_graph
    from app.core.search_index import SearchIndex
    from app.core.text_embedder import embed_text
    from app.core.uvnet_embedder import embed

    idx = SearchIndex(db_path=tmp / "parts.db")
    for stem, meta_text in [
        ("partA", "aluminum bracket CNC milling M6 holes Acme"),
        ("partB", "stainless steel shaft turning"),
        ("partC", "titanium plate milling aerospace"),
    ]:
        graph = step_to_dgl_graph(FIXTURE_DIR / f"{stem}.step")
        geo_vec, hist = embed(graph)
        text_vec = embed_text(meta_text)
        idx.add(
            name=stem,
            geo_vec=geo_vec.numpy(),
            text_vec=text_vec,
            material="Aluminum" if "aluminum" in meta_text else "Steel",
            histogram=hist,
        )
    idx.save(geo_path=tmp / "geo.faiss", text_path=tmp / "text.faiss")

    # Patch singleton before importing app
    import app.core.index_singleton as singleton_mod

    singleton_mod._index = idx

    from app.main import app

    return TestClient(app)


def test_health(seeded_client):
    r = seeded_client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["index_size"] == 3


def test_search_cad_returns_ranked(seeded_client):
    step_bytes = (FIXTURE_DIR / "partA.step").read_bytes()
    r = seeded_client.post(
        "/api/search/cad",
        files={"file": ("partA.step", step_bytes, "application/octet-stream")},
    )
    assert r.status_code == 200
    body = r.json()
    results = body["results"]
    assert len(results) > 0
    assert {"geo_score", "text_score", "badge", "name"} <= results[0].keys()
    # self-search should be #1
    assert results[0]["name"] == "partA"
    assert results[0]["geo_score"] > 0.99
    # scores descending
    scores = [res["final_score"] for res in results]
    assert scores == sorted(scores, reverse=True)


def test_search_cad_wrong_extension(seeded_client):
    r = seeded_client.post(
        "/api/search/cad",
        files={"file": ("model.obj", b"fake", "application/octet-stream")},
    )
    assert r.status_code == 400


def test_text_only_skips_uvnet(seeded_client):
    r = seeded_client.post("/api/search/text", json={"q": "aluminum bracket CNC"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) > 0
    assert body["results"][0]["name"] == "partA"


def test_text_empty_query(seeded_client):
    r = seeded_client.post("/api/search/text", json={"q": "   "})
    assert r.status_code == 400


def test_library_parts_list(seeded_client):
    r = seeded_client.get("/api/library/parts")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    names = {p["name"] for p in body["parts"]}
    assert names == {"partA", "partB", "partC"}


def test_library_part_detail(seeded_client):
    r = seeded_client.get("/api/library/parts/1")
    assert r.status_code == 200
    body = r.json()
    assert "histogram" in body and "occ_stats" in body


def test_library_part_not_found(seeded_client):
    r = seeded_client.get("/api/library/parts/9999")
    assert r.status_code == 404
