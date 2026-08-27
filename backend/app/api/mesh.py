"""GET /api/mesh/{id} — glTF export from stored STEP.
GET /api/thumbnail/{id} — placeholder PNG (full render at Phase 8).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import DATA_DIR
from app.core.index_singleton import get_index

router = APIRouter()

_MESH_DIR = DATA_DIR / "meshes"
_THUMB_DIR = DATA_DIR / "thumbnails"


def _export_gltf(step_path: Path, out_path: Path) -> None:
    """Tessellate a STEP and write a glb file via pythonOCC."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.DataExchange import write_gltf_file

    from app.occwl.io import load_shell

    solids = load_shell(str(step_path))
    if not solids:
        raise ValueError("No solids loaded")
    shape = solids[0].topods_shape()
    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5).Perform()
    write_gltf_file(shape, str(out_path))


@router.get("/mesh/{part_id}")
def get_mesh(part_id: int):
    idx = get_index()
    part = idx._fetch_part(part_id)
    if part is None:
        raise HTTPException(404, f"Part {part_id} not found")

    # Serve cached glb if already exported
    cached = _MESH_DIR / f"{part_id}.glb"
    if cached.exists():
        return FileResponse(str(cached), media_type="model/gltf-binary")

    # No cached file — return 404 with helpful message
    # (glTF export happens at index time in index_library.py; this is the fallback)
    raise HTTPException(
        404,
        "Mesh not yet generated for this part. Re-index with index_library.py to build glTF files.",
    )


@router.get("/thumbnail/{part_id}")
def get_thumbnail(part_id: int):
    idx = get_index()
    part = idx._fetch_part(part_id)
    if part is None:
        raise HTTPException(404, f"Part {part_id} not found")

    cached = _THUMB_DIR / f"{part_id}.png"
    if cached.exists():
        return FileResponse(str(cached), media_type="image/png")

    raise HTTPException(404, "Thumbnail not yet generated for this part.")
