import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "app" / "data"
MODELS_DIR = BASE_DIR / "app" / "models"

# ── Search weights & thresholds ────────────────────────────────────────────────
GEO_WEIGHT = float(os.getenv("GEO_WEIGHT", "0.7"))
TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.3"))
TOP_K = int(os.getenv("TOP_K", "5"))
DUP_THRESHOLD = float(os.getenv("DUP_THRESHOLD", "0.95"))  # geo cosine → Near-duplicate badge
WEAK_THRESHOLD = float(os.getenv("WEAK_THRESHOLD", "0.55"))  # max score below → Weak-match badge

# ── LLM ────────────────────────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none")  # "anthropic" | "gemini" | "none"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ── Upload ──────────────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
ALLOWED_EXTENSIONS = {".step", ".stp"}

# ── CORS ────────────────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
