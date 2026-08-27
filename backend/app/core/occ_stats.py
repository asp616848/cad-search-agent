"""Geometric properties extracted via pythonOCC for a loaded solid shape."""

from __future__ import annotations

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer


def _count_shapes(shape, shape_type: int) -> int:
    explorer = TopExp_Explorer(shape, shape_type)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def compute(shape) -> dict:
    """Return geometric stats for a pythonOCC TopoDS_Shape (solid)."""
    props = GProp_GProps()

    brepgprop.VolumeProperties(shape, props)
    volume = props.Mass()

    brepgprop.SurfaceProperties(shape, props)
    surface_area = props.Mass()

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bx = xmax - xmin
    by = ymax - ymin
    bz = zmax - zmin

    bbox_vol = bx * by * bz
    solidity = volume / bbox_vol if bbox_vol > 1e-12 else 0.0

    face_count = _count_shapes(shape, TopAbs_FACE)
    edge_count = _count_shapes(shape, TopAbs_EDGE)

    return {
        "face_count": face_count,
        "edge_count": edge_count,
        "volume": volume,
        "surface_area": surface_area,
        "bbox_x": bx,
        "bbox_y": by,
        "bbox_z": bz,
        "solidity": min(max(solidity, 0.0), 1.0),
    }
