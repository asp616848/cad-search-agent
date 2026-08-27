"""Phase 6 gate — glTF mesh endpoint."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"
DATA_DIR = Path(__file__).parent.parent.parent / "app" / "data"
MESH_DIR = DATA_DIR / "meshes"


@pytest.fixture(scope="module")
def mesh_client(tmp_path_factory):
    """Client with a seeded index that has real glb files."""
    from app.core.graph_builder import step_to_dgl_graph
    from app.core.search_index import SearchIndex
    from app.core.text_embedder import embed_text
    from app.core.uvnet_embedder import embed

    tmp = tmp_path_factory.mktemp("mesh_data")
    mesh_dir = tmp / "meshes"
    mesh_dir.mkdir()

    idx = SearchIndex(db_path=tmp / "parts.db")

    # Build index and export glTF
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.DataExchange import write_gltf_file

    from app.occwl.io import load_shell

    part_ids = []
    for stem in ("partA", "partB", "partC"):
        step_path = FIXTURE_DIR / f"{stem}.step"
        graph = step_to_dgl_graph(step_path)
        geo_vec, hist = embed(graph)
        text_vec = embed_text(stem)
        pid = idx.add(name=stem, geo_vec=geo_vec.numpy(), text_vec=text_vec, histogram=hist)
        part_ids.append(pid)

        # Export glb
        solids = load_shell(str(step_path))
        shape = solids[0].topods_shape()
        BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5).Perform()
        write_gltf_file(shape, str(mesh_dir / f"{pid}.glb"))

    idx.save(geo_path=tmp / "geo.faiss", text_path=tmp / "text.faiss")

    import app.core.index_singleton as singleton_mod

    singleton_mod._index = idx

    # Patch DATA_DIR so mesh endpoint finds the right directory
    import app.api.mesh as mesh_mod

    mesh_mod._MESH_DIR = mesh_dir

    from app.main import app

    return TestClient(app), part_ids


def test_mesh_bytes_valid(mesh_client):
    client, part_ids = mesh_client
    r = client.get(f"/api/mesh/{part_ids[0]}")
    assert r.status_code == 200
    b = r.content
    # glTF binary magic or JSON gltf
    assert b[:4] == b"glTF" or b.startswith(b"{")


def test_mesh_not_found(mesh_client):
    client, _ = mesh_client
    r = client.get("/api/mesh/99999")
    assert r.status_code == 404


def test_thumbnail_not_found_graceful(mesh_client):
    client, part_ids = mesh_client
    r = client.get(f"/api/thumbnail/{part_ids[0]}")
    assert r.status_code == 404
