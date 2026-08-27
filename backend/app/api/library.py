"""Library management endpoints."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.config import ALLOWED_EXTENSIONS, DATA_DIR, MAX_UPLOAD_MB
from app.core.index_singleton import get_index

router = APIRouter()

_MAX_BYTES = MAX_UPLOAD_MB * 1024 * 1024


@router.get("/library/parts")
def list_parts(
    material: str | None = Query(default=None),
    process: str | None = Query(default=None),
):
    idx = get_index()
    query = "SELECT id,name,material,process,cost,supplier,thumb_path FROM parts WHERE 1=1"
    params: list = []
    if material:
        query += " AND material LIKE ?"
        params.append(f"%{material}%")
    if process:
        query += " AND process LIKE ?"
        params.append(f"%{process}%")
    query += " ORDER BY id"
    rows = idx._conn.execute(query, params).fetchall()
    return {
        "parts": [
            {
                "id": r[0],
                "name": r[1],
                "material": r[2],
                "process": r[3],
                "cost": r[4],
                "supplier": r[5],
                "thumb_path": r[6],
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/library/parts/{part_id}")
def get_part(part_id: int):
    idx = get_index()
    part = idx._fetch_part(part_id)
    if part is None:
        raise HTTPException(404, f"Part {part_id} not found")
    return {
        "id": part.id,
        "name": part.name,
        "material": part.material,
        "process": part.process,
        "cost": part.cost,
        "supplier": part.supplier,
        "notes": part.notes,
        "known_issues": part.known_issues,
        "ppap_notes": part.ppap_notes,
        "histogram": part.histogram,
        "occ_stats": part.occ_stats,
        "mesh_url": f"/api/mesh/{part_id}" if part.mesh_path else None,
        "thumb_url": f"/api/thumbnail/{part_id}" if part.thumb_path else None,
    }


@router.post("/library/index")
async def index_part(
    file: UploadFile = File(...),
    metadata: str = Form(default="{}"),
):
    """Add a new part to the live index without restarting."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {ALLOWED_EXTENSIONS} accepted")

    content = await file.read()
    if len(content) > _MAX_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB limit")

    try:
        meta = json.loads(metadata)
    except json.JSONDecodeError:
        raise HTTPException(400, "metadata must be valid JSON")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        from app.core.graph_builder import step_to_dgl_graph
        from app.core.occ_stats import compute as occ_stats
        from app.core.text_embedder import embed_text, histogram_to_text
        from app.core.uvnet_embedder import embed
        from app.occwl.io import load_shell

        try:
            graph = step_to_dgl_graph(tmp_path)
            geo_vec, histogram = embed(graph)
        except Exception as e:
            raise HTTPException(422, str(e)) from e

        solids = load_shell(str(tmp_path))
        stats = occ_stats(solids[0].topods_shape()) if solids else {}

        text_parts = [
            meta.get("name", file.filename or ""),
            meta.get("material", ""),
            meta.get("process", ""),
            meta.get("notes", ""),
            meta.get("known_issues", ""),
            histogram_to_text(histogram),
        ]
        text_vec = embed_text(" ".join(p for p in text_parts if p))

        idx = get_index()
        part_id = idx.add(
            name=meta.get("name", file.filename or tmp_path.stem),
            geo_vec=geo_vec.numpy(),
            text_vec=text_vec,
            material=meta.get("material", ""),
            process=meta.get("process", ""),
            cost=float(meta.get("cost", 0)),
            supplier=meta.get("supplier", ""),
            notes=meta.get("notes", ""),
            known_issues=meta.get("known_issues", ""),
            ppap_notes=meta.get("ppap_notes", ""),
            histogram=histogram,
            occ_stats=stats,
        )
        idx.save()

        from app.api.mesh import _export_gltf
        from app.core.thumbnail import render_thumbnail

        glb_path = DATA_DIR / "meshes" / f"{part_id}.glb"
        try:
            _export_gltf(tmp_path, glb_path)
            render_thumbnail(glb_path, DATA_DIR / "thumbnails" / f"{part_id}.png")
        except Exception:
            pass  # mesh/thumbnail are best-effort; part is still indexed

        return {"id": part_id, "name": meta.get("name", ""), "index_size": idx.count()}
    finally:
        tmp_path.unlink(missing_ok=True)
