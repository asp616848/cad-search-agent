"""Static PNG thumbnail renderer.

Renders an already-exported glb with a small self-contained software
z-buffer rasterizer instead of a GPU/window-server-backed library. This
was deliberately chosen after two dead ends on macOS:
  - trimesh's pyglet-based scene.save_image() crashes on window teardown
    (pyglet 1.5 Cocoa event-loop bug).
  - pyvista/VTK offscreen rendering hangs waiting for a real window
    server that isn't available in a headless/CI context.
matplotlib's Poly3DCollection was also tried and rejected: it sorts
whole polygons by centroid depth (painter's algorithm), which produces
a garbled, uninterpretable image for any mesh with concave features or
overlapping geometry — exactly what machined parts with slots/pockets are.
A real per-pixel z-buffer (below) renders correctly regardless of
concavity and has zero GPU/display dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

_BG = (18, 22, 43)  # ink-900, matches the interactive viewer background
_LIGHT_DIR = np.array([0.4, 0.6, 0.7])
_LIGHT_DIR = _LIGHT_DIR / np.linalg.norm(_LIGHT_DIR)
_BASE_COLOR = np.array([201, 206, 217])  # light gray, matches result card chips
_AMBIENT = 0.35


def _load_triangles(glb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices Nx3, faces Mx3 int) as a single triangulated mesh."""
    loaded = trimesh.load(str(glb_path), force="mesh")
    mesh = loaded if isinstance(loaded, trimesh.Trimesh) else loaded.dump(concatenate=True)
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("Mesh has no geometry to render")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _isometric_rotation() -> np.ndarray:
    """Rotation matrix for a classic ~isometric CAD viewing angle."""
    az, el = np.radians(45.0), np.radians(-30.0)
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]])
    return rx @ rz


def _rasterize(verts: np.ndarray, faces: np.ndarray, size: int) -> np.ndarray:
    rot = _isometric_rotation()
    view = verts @ rot.T  # camera-space coords; view[:,2] increases toward viewer

    center = view.mean(axis=0)
    view -= center
    radius = max(np.abs(view).max(), 1e-6)
    scale = (size * 0.42) / radius

    # Orthographic projection: x -> screen col, y -> screen row (flip), z -> depth
    px = view[:, 0] * scale + size / 2
    py = -view[:, 1] * scale + size / 2
    depth = view[:, 2]

    color_buf = np.tile(np.array(_BG, dtype=np.float64), (size, size, 1))
    z_buf = np.full((size, size), -np.inf)

    # Per-face flat shading from the (unrotated) face normal so lighting
    # stays fixed relative to the camera regardless of part orientation.
    tri_verts = verts[faces]
    edge1 = tri_verts[:, 1] - tri_verts[:, 0]
    edge2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals = np.cross(edge1, edge2)
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals @ rot.T / norm_len
    intensity = np.clip(normals @ _LIGHT_DIR, 0, 1)
    shade = _AMBIENT + (1 - _AMBIENT) * intensity
    face_colors = np.clip(_BASE_COLOR[None, :] * shade[:, None], 0, 255)

    for i in range(faces.shape[0]):
        a, b, c = faces[i]
        x0, y0, x1, y1, x2, y2 = px[a], py[a], px[b], py[b], px[c], py[c]

        min_x = max(int(np.floor(min(x0, x1, x2))), 0)
        max_x = min(int(np.ceil(max(x0, x1, x2))), size - 1)
        min_y = max(int(np.floor(min(y0, y1, y2))), 0)
        max_y = min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if min_x > max_x or min_y > max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue  # degenerate (zero-area) triangle

        ys, xs = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
        xs = xs.astype(np.float64) + 0.5
        ys = ys.astype(np.float64) + 0.5

        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        region_z = z_buf[min_y : max_y + 1, min_x : max_x + 1]
        closer = inside & (z > region_z)
        if not closer.any():
            continue

        region_z[closer] = z[closer]
        z_buf[min_y : max_y + 1, min_x : max_x + 1] = region_z
        color_buf[min_y : max_y + 1, min_x : max_x + 1][closer] = face_colors[i]

    return color_buf.astype(np.uint8)


def render_thumbnail(glb_path: Path, out_path: Path, size: int = 320) -> None:
    """Render a glb to a PNG file with a real per-pixel z-buffer.

    Raises on failure — caller decides whether that's fatal (index_library.py
    treats it as best-effort, same as glTF export).
    """
    verts, faces = _load_triangles(glb_path)
    image = _rasterize(verts, faces, size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(str(out_path))


def render_thumbnail_bytes(glb_path: Path, size: int = 320) -> bytes:
    """Same as render_thumbnail but returns PNG bytes without touching disk."""
    import io

    verts, faces = _load_triangles(glb_path)
    image = _rasterize(verts, faces, size)
    buf = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
