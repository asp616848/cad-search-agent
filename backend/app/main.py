from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, library, mesh, search
from app.config import CORS_ORIGINS

app = FastAPI(title="cad-search-agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(library.router, prefix="/api")
app.include_router(mesh.router, prefix="/api")


@app.get("/")
def root():
    return {"service": "cad-search-agent", "status": "ok"}
