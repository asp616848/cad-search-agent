from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.index_singleton import get_index

router = APIRouter()

_started_at = datetime.now(timezone.utc).isoformat()


@router.get("/health")
def health():
    idx = get_index()
    try:
        from app.core.uvnet_embedder import get_model

        get_model()
        model_loaded = True
    except Exception:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "index_size": idx.count(),
        "started_at": _started_at,
    }
