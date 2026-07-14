"""Central configuration, loaded from environment variables / .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
JOBS_PATH = DATA_DIR / "jobs.json"
# LangChain's FAISS integration writes these two files with ``jobs`` as the
# index name. ``jobs.json`` remains a small, human-readable export of metadata.
FAISS_INDEX_NAME = "jobs"
FAISS_INDEX_PATH = DATA_DIR / f"{FAISS_INDEX_NAME}.faiss"
FAISS_DOCSTORE_PATH = DATA_DIR / f"{FAISS_INDEX_NAME}.pkl"

# --- Gemini --------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001").strip()

# --- Data source ---------------------------------------------------------
REMOTIVE_API = "https://remotive.com/api/remote-jobs"
MAX_JOBS = int(os.getenv("MAX_JOBS", "800"))

# --- Server --------------------------------------------------------------
PORT = int(os.getenv("PORT", "7860"))
