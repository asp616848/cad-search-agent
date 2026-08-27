"""Dual FAISS index + SQLite store + score fusion."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.config import DATA_DIR, DUP_THRESHOLD, GEO_WEIGHT, TEXT_WEIGHT, TOP_K

_DB_PATH = DATA_DIR / "parts.db"
_GEO_INDEX_PATH = DATA_DIR / "geo.faiss"
_TEXT_INDEX_PATH = DATA_DIR / "text.faiss"

GEO_DIM = 128
TEXT_DIM = 384


@dataclass
class PartRecord:
    id: int
    name: str
    material: str = ""
    process: str = ""
    cost: float = 0.0
    supplier: str = ""
    notes: str = ""
    known_issues: str = ""
    ppap_notes: str = ""
    histogram: dict[str, int] = field(default_factory=dict)
    occ_stats: dict[str, Any] = field(default_factory=dict)
    mesh_path: str = ""
    thumb_path: str = ""


@dataclass
class SearchHit:
    part: PartRecord
    geo_score: float = 0.0
    text_score: float = 0.0
    final_score: float = 0.0
    is_duplicate: bool = False


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS parts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT NOT NULL,
    material     TEXT DEFAULT '',
    process      TEXT DEFAULT '',
    cost         REAL DEFAULT 0,
    supplier     TEXT DEFAULT '',
    notes        TEXT DEFAULT '',
    known_issues TEXT DEFAULT '',
    ppap_notes   TEXT DEFAULT '',
    histogram_json  TEXT DEFAULT '{}',
    occ_stats_json  TEXT DEFAULT '{}',
    mesh_path    TEXT DEFAULT '',
    thumb_path   TEXT DEFAULT '',
    geo_vec_blob BLOB,
    text_vec_blob BLOB
);
"""


class SearchIndex:
    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(_CREATE_SQL)
        self._conn.commit()
        self.geo_index = faiss.IndexFlatIP(GEO_DIM)
        self.text_index = faiss.IndexFlatIP(TEXT_DIM)
        self._faiss_id_to_part_id: list[int] = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        geo_vec: np.ndarray,
        text_vec: np.ndarray,
        material: str = "",
        process: str = "",
        cost: float = 0.0,
        supplier: str = "",
        notes: str = "",
        known_issues: str = "",
        ppap_notes: str = "",
        histogram: dict | None = None,
        occ_stats: dict | None = None,
        mesh_path: str = "",
        thumb_path: str = "",
    ) -> int:
        """Insert a part and add its vectors to both FAISS indexes. Returns part id."""
        histogram = histogram or {}
        occ_stats = occ_stats or {}
        geo_vec = _unit(geo_vec)
        text_vec = _unit(text_vec)

        cur = self._conn.execute(
            """
            INSERT INTO parts
              (name, material, process, cost, supplier, notes, known_issues, ppap_notes,
               histogram_json, occ_stats_json, mesh_path, thumb_path, geo_vec_blob, text_vec_blob)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                name,
                material,
                process,
                cost,
                supplier,
                notes,
                known_issues,
                ppap_notes,
                json.dumps(histogram),
                json.dumps(occ_stats),
                mesh_path,
                thumb_path,
                geo_vec.astype(np.float32).tobytes(),
                text_vec.astype(np.float32).tobytes(),
            ),
        )
        self._conn.commit()
        part_id = cur.lastrowid

        self.geo_index.add(geo_vec.reshape(1, -1).astype(np.float32))
        self.text_index.add(text_vec.reshape(1, -1).astype(np.float32))
        self._faiss_id_to_part_id.append(part_id)
        return part_id

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def _fetch_part(self, part_id: int) -> PartRecord | None:
        row = self._conn.execute(
            "SELECT id,name,material,process,cost,supplier,notes,known_issues,"
            "ppap_notes,histogram_json,occ_stats_json,mesh_path,thumb_path "
            "FROM parts WHERE id=?",
            (part_id,),
        ).fetchone()
        if row is None:
            return None
        return PartRecord(
            id=row[0],
            name=row[1],
            material=row[2],
            process=row[3],
            cost=row[4],
            supplier=row[5],
            notes=row[6],
            known_issues=row[7],
            ppap_notes=row[8],
            histogram=json.loads(row[9]),
            occ_stats=json.loads(row[10]),
            mesh_path=row[11],
            thumb_path=row[12],
        )

    def search_cad(self, geo_vec: np.ndarray, k: int = TOP_K) -> list[SearchHit]:
        """Return top-k hits ranked by geo cosine score."""
        if self.geo_index.ntotal == 0:
            return []
        q = _unit(geo_vec).reshape(1, -1).astype(np.float32)
        scores, indices = self.geo_index.search(q, min(k, self.geo_index.ntotal))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            part_id = self._faiss_id_to_part_id[idx]
            part = self._fetch_part(part_id)
            if part is None:
                continue
            hits.append(
                SearchHit(
                    part=part,
                    geo_score=float(score),
                    final_score=float(score),
                    is_duplicate=float(score) >= DUP_THRESHOLD,
                )
            )
        return hits

    def search_text(self, text_vec: np.ndarray, k: int = TOP_K) -> list[SearchHit]:
        """Return top-k hits ranked by text cosine score."""
        if self.text_index.ntotal == 0:
            return []
        q = _unit(text_vec).reshape(1, -1).astype(np.float32)
        scores, indices = self.text_index.search(q, min(k, self.text_index.ntotal))
        hits = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            part_id = self._faiss_id_to_part_id[idx]
            part = self._fetch_part(part_id)
            if part is None:
                continue
            hits.append(
                SearchHit(
                    part=part,
                    text_score=float(score),
                    final_score=float(score),
                )
            )
        return hits

    def count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM parts").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        geo_path: Path = _GEO_INDEX_PATH,
        text_path: Path = _TEXT_INDEX_PATH,
    ) -> None:
        geo_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.geo_index, str(geo_path))
        faiss.write_index(self.text_index, str(text_path))

    @classmethod
    def load(
        cls,
        db_path: Path = _DB_PATH,
        geo_path: Path = _GEO_INDEX_PATH,
        text_path: Path = _TEXT_INDEX_PATH,
    ) -> "SearchIndex":
        idx = cls(db_path=db_path)
        if geo_path.exists():
            idx.geo_index = faiss.read_index(str(geo_path))
        if text_path.exists():
            idx.text_index = faiss.read_index(str(text_path))
        # rebuild faiss_id → part_id map from DB insertion order
        rows = idx._conn.execute("SELECT id FROM parts ORDER BY id").fetchall()
        idx._faiss_id_to_part_id = [r[0] for r in rows]
        return idx


# ------------------------------------------------------------------
# Score fusion (standalone — easy to unit test)
# ------------------------------------------------------------------


def fuse(
    geo_scores: dict[int, float],
    text_scores: dict[int, float],
    geo_weight: float = GEO_WEIGHT,
    text_weight: float = TEXT_WEIGHT,
) -> dict[int, float]:
    """
    Combine geo and text cosine scores into a single ranked dict.

    Args:
        geo_scores:  {part_id: cosine_score}
        text_scores: {part_id: cosine_score}
        geo_weight:  weight for geo score (default from config)
        text_weight: weight for text score (default from config)

    Returns:
        {part_id: final_score} sorted descending by final_score
    """
    all_ids = set(geo_scores) | set(text_scores)
    result = {}
    for pid in all_ids:
        g = geo_scores.get(pid, 0.0)
        t = text_scores.get(pid, 0.0)
        result[pid] = geo_weight * g + text_weight * t
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)
