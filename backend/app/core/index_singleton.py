"""Global SearchIndex singleton loaded once at startup."""

from __future__ import annotations

from app.config import DATA_DIR
from app.core.search_index import SearchIndex

_index: SearchIndex | None = None


def get_index() -> SearchIndex:
    global _index
    if _index is None:
        _index = SearchIndex.load(
            db_path=DATA_DIR / "parts.db",
            geo_path=DATA_DIR / "geo.faiss",
            text_path=DATA_DIR / "text.faiss",
        )
    return _index
