"""Central configuration, loaded from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"

# --- Gemini --------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip()

# --- Data source ---------------------------------------------------------
REMOTIVE_API = "https://remotive.com/api/remote-jobs"
MAX_JOBS = int(os.getenv("MAX_JOBS", "800"))

# --- Server --------------------------------------------------------------
PORT = int(os.getenv("PORT", "7860"))
