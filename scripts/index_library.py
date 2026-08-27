"""Index a directory of STEP files + metadata JSON into FAISS + SQLite.

Usage:
  python scripts/index_library.py \
    --step_dir data/demo_steps \
    --metadata data/demo_metadata.json \
    --output backend/app/data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from repo root or backend/
ROOT = Path(__file__).parent.parent
BACKEND = ROOT / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.core.graph_builder import step_to_dgl_graph  # noqa: E402
from app.core.occ_stats import compute as occ_stats  # noqa: E402
from app.core.search_index import SearchIndex  # noqa: E402
from app.core.text_embedder import embed_text  # noqa: E402
from app.core.thumbnail import render_thumbnail  # noqa: E402
from app.core.uvnet_embedder import embed  # noqa: E402
from app.occwl.io import load_shell  # noqa: E402


def _build_text_doc(meta: dict) -> str:
    """Concatenate metadata fields into a single search document."""
    parts = [
        meta.get("name", ""),
        meta.get("material", ""),
        meta.get("process", ""),
        meta.get("supplier", ""),
        meta.get("notes", ""),
        meta.get("known_issues", ""),
        meta.get("ppap_notes", ""),
    ]
    return " ".join(p for p in parts if p)


def _export_gltf(step_path: Path, out_path: Path) -> None:
    """Tessellate STEP and export as glb. Raises on failure."""
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Extend.DataExchange import write_gltf_file

    solids = load_shell(str(step_path))
    if not solids:
        raise ValueError("No solids loaded")
    shape = solids[0].topods_shape()
    BRepMesh_IncrementalMesh(shape, 0.1, False, 0.5).Perform()
    write_gltf_file(shape, str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step_dir", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    step_dir: Path = args.step_dir
    meta_path: Path = args.metadata
    out_dir: Path = args.output

    if not step_dir.exists():
        print(f"ERROR: step_dir not found: {step_dir}", file=sys.stderr)
        sys.exit(1)
    if not meta_path.exists():
        print(f"ERROR: metadata file not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_by_filename: dict[str, dict] = {}
    with open(meta_path) as f:
        for entry in json.load(f):
            meta_by_filename[entry["filename"]] = entry

    step_files = sorted(step_dir.glob("*.step")) + sorted(step_dir.glob("*.stp"))
    print(f"Found {len(step_files)} STEP files in {step_dir}")

    mesh_dir = out_dir / "meshes"
    mesh_dir.mkdir(exist_ok=True)
    thumb_dir = out_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)

    idx = SearchIndex(db_path=out_dir / "parts.db")
    succeeded, failed = [], []

    for step_path in step_files:
        fname = step_path.name
        meta = meta_by_filename.get(fname, {"filename": fname, "name": step_path.stem})
        t0 = time.time()
        try:
            graph = step_to_dgl_graph(step_path)
            geo_vec, histogram = embed(graph)

            solids = load_shell(str(step_path))
            stats = occ_stats(solids[0].topods_shape()) if solids else {}

            text_doc = _build_text_doc(meta)
            text_vec = embed_text(text_doc)

            part_id = idx.add(
                name=meta.get("name", step_path.stem),
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

            # glTF export (best-effort — failure does not abort indexing)
            glb_path = mesh_dir / f"{part_id}.glb"
            try:
                _export_gltf(step_path, glb_path)
            except Exception as e:
                print(f"  WARN  glTF export failed for {fname}: {e}")

            # Thumbnail render (best-effort, depends on the glb existing)
            if glb_path.exists():
                try:
                    render_thumbnail(glb_path, thumb_dir / f"{part_id}.png")
                except Exception as e:
                    print(f"  WARN  thumbnail render failed for {fname}: {e}")

            elapsed = time.time() - t0
            print(f"  OK  {fname} ({elapsed:.1f}s)")
            succeeded.append(fname)
        except Exception as e:
            print(f"  FAIL {fname}: {e}")
            failed.append((fname, str(e)))

    idx.save(
        geo_path=out_dir / "geo.faiss",
        text_path=out_dir / "text.faiss",
    )
    idx.close()

    print("\n=== Library card ===")
    print(f"Indexed: {len(succeeded)}  Failed: {len(failed)}")
    if failed:
        print("Failed files:")
        for fname, err in failed:
            print(f"  {fname}: {err}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
