"""Build a UV-Net-ready DGL graph from a STEP file."""

from __future__ import annotations

import logging
from pathlib import Path

import dgl
import numpy as np
import torch
from dgl import DGLGraph

from app.occwl.graph import face_adjacency
from app.occwl.io import load_shell
from app.occwl.uvgrid import ugrid, uvgrid

logger = logging.getLogger(__name__)

_DEFAULT_CURV_U = 10
_DEFAULT_SURF_U = 10
_DEFAULT_SURF_V = 10


def _center_and_scale(
    inp: torch.Tensor, return_center_scale: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, float]:
    """Center XYZ channels on the visible-point bbox and scale so the diagonal max is 2."""
    pts = inp[..., :3].reshape(-1, 3)
    mask = inp[..., 6].reshape(-1)
    pts_inside = pts[mask == 1]
    if pts_inside.numel() == 0:
        raise ValueError("Cannot center/scale UV-grid: no visible sample points (mask == 1)")
    bbox_min = pts_inside.min(dim=0).values
    bbox_max = pts_inside.max(dim=0).values
    center = 0.5 * (bbox_min + bbox_max)
    diag = bbox_max - bbox_min
    scale = 2.0 / max(diag[0].item(), diag[1].item(), diag[2].item(), 1e-6)
    inp = inp.clone()
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp


def _build_feature_arrays(
    solid: object,
    curv_num_u_samples: int = _DEFAULT_CURV_U,
    surf_num_u_samples: int = _DEFAULT_SURF_U,
    surf_num_v_samples: int = _DEFAULT_SURF_V,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int], int]:
    """Extract face UV-grids and edge U-grids from a face-adjacency graph."""
    graph = face_adjacency(solid)
    if graph is None:
        raise ValueError("Failed to build face-adjacency graph (non-manifold or open shape)")

    n_nodes = len(graph.nodes)
    if n_nodes == 0:
        raise ValueError("Face-adjacency graph is empty")

    graph_face_feat: list[np.ndarray] = []
    for face_idx in sorted(graph.nodes):
        face = graph.nodes[face_idx]["face"]
        points = uvgrid(face, method="point", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        normals = uvgrid(face, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples)
        visibility_status = uvgrid(
            face, method="visibility_status", num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
        if points is None or normals is None or visibility_status is None:
            raise ValueError(f"UV-grid sampling failed for face {face_idx}")
        mask = np.logical_or(visibility_status == 0, visibility_status == 2)
        face_feat = np.concatenate((points, normals, mask), axis=-1)
        graph_face_feat.append(face_feat)
    graph_face_feat_arr = np.asarray(graph_face_feat, dtype=np.float32)
    if graph_face_feat_arr.size == 0:
        raise ValueError("No face UV-grid features were produced")

    edges = list(graph.edges)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]

    graph_edge_feat: list[np.ndarray] = []
    valid_src: list[int] = []
    valid_dst: list[int] = []
    for edge_idx, s, d in zip(edges, src, dst):
        edge = graph.edges[edge_idx]["edge"]
        if not edge.has_curve():
            # Degenerate edges (e.g. cone apex) have no curve — skip, same as upstream
            continue
        points = ugrid(edge, method="point", num_u=curv_num_u_samples)
        tangents = ugrid(edge, method="tangent", num_u=curv_num_u_samples)
        if points is None or tangents is None:
            raise ValueError(f"U-grid sampling failed for edge {edge_idx}")
        edge_feat = np.concatenate((points, tangents), axis=-1)
        graph_edge_feat.append(edge_feat)
        valid_src.append(s)
        valid_dst.append(d)

    src[:] = valid_src
    dst[:] = valid_dst

    if not graph_edge_feat:
        raise ValueError("Face-adjacency graph has no valid (non-degenerate) edges")
    graph_edge_feat_arr = np.asarray(graph_edge_feat, dtype=np.float32)
    return graph_face_feat_arr, graph_edge_feat_arr, src, dst, n_nodes


def step_to_dgl_graph(step_path: str | Path) -> DGLGraph:
    """Load a STEP file and return a preprocessed DGL graph ready for UV-Net inference."""
    path = Path(step_path)
    if not path.exists():
        raise FileNotFoundError(f"STEP file not found: {path}")
    if not path.is_file():
        raise ValueError(f"STEP path is not a file: {path}")

    solids = load_shell(str(path))
    if not solids:
        raise ValueError(f"Could not load any solids from STEP file: {path}")
    if len(solids) > 1:
        logger.warning(
            "STEP file %s contains %d solids; using the first only",
            path,
            len(solids),
        )
    solid = solids[0]

    face_feat, edge_feat, src, dst, n_nodes = _build_feature_arrays(solid)
    if n_nodes == 0 or face_feat.shape[0] == 0:
        raise ValueError(f"Empty graph produced from STEP file: {path}")

    dgl_graph = dgl.graph((src, dst), num_nodes=n_nodes)
    dgl_graph.ndata["x"] = torch.tensor(face_feat, dtype=torch.float32)
    dgl_graph.edata["x"] = torch.tensor(edge_feat, dtype=torch.float32)

    dgl_graph.ndata["x"], center, scale = _center_and_scale(
        dgl_graph.ndata["x"], return_center_scale=True
    )
    dgl_graph.edata["x"][..., :3] -= center
    dgl_graph.edata["x"][..., :3] *= scale

    dgl_graph.ndata["x"] = dgl_graph.ndata["x"].permute(0, 3, 1, 2).float()
    dgl_graph.edata["x"] = dgl_graph.edata["x"].permute(0, 2, 1).float()
    return dgl_graph
