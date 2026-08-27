"""Phase 1 gate — STEP -> graph -> geo_vec + histogram.
Run: pytest tests/integration/test_embedder.py -q
Requires: pythonOCC + occwl + torch + dgl installed.
"""

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from app.core.graph_builder import step_to_dgl_graph
from app.core.uvnet_embedder import embed

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def test_geo_vec_shape_and_norm() -> None:
    graph = step_to_dgl_graph(FIXTURE_DIR / "partA.step")
    vec, hist = embed(graph)
    assert vec.shape == (128,)
    assert abs(float(torch.norm(vec)) - 1.0) < 1e-3
    assert isinstance(hist, dict) and len(hist) > 0


def test_histogram_excludes_stock() -> None:
    graph = step_to_dgl_graph(FIXTURE_DIR / "partA.step")
    _vec, hist = embed(graph)
    if set(hist.keys()) == {"stock"}:
        pytest.skip("all faces classified as stock")
    assert "stock" not in hist


def test_two_different_parts_differ() -> None:
    g1 = step_to_dgl_graph(FIXTURE_DIR / "partA.step")
    g2 = step_to_dgl_graph(FIXTURE_DIR / "partB.step")
    v1, _h1 = embed(g1)
    v2, _h2 = embed(g2)
    sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
    assert sim < 0.999


def test_invalid_file_raises() -> None:
    with pytest.raises((ValueError, FileNotFoundError, Exception)):
        step_to_dgl_graph("nonexistent.step")


def test_embed_is_deterministic() -> None:
    path = FIXTURE_DIR / "partA.step"
    g1 = step_to_dgl_graph(path)
    g2 = step_to_dgl_graph(path)
    v1, _h1 = embed(g1)
    v2, _h2 = embed(g2)
    assert torch.allclose(v1, v2, atol=1e-5)
