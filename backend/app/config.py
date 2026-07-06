import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


BASE_DIR = Path(__file__).resolve().parents[2]

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")
VECTOR_STORE_PATH = os.getenv(
    "VECTOR_STORE_PATH",
    str(BASE_DIR / "data" / "vector_store"),
)
MAX_RETRIEVAL_RETRIES = int(os.getenv("MAX_RETRIEVAL_RETRIES", "2"))
TOP_K = int(os.getenv("TOP_K", "1"))
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
