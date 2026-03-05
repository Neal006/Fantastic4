import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------- LLM ----------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")

# ---------- RAG ----------
INVERTER_MANUAL_PATH = os.getenv(
    "INVERTER_MANUAL_PATH",
    str(BASE_DIR / "2cdb179b-9321-4b69-871a-ff5f5df3b3ef.pdf"),
)
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "vector_store"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

# ---------- Outputs ----------
TICKET_DIR = os.getenv("TICKET_DIR", str(BASE_DIR / "tickets"))
