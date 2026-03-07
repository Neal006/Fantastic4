"""
Configuration for the comparative analysis / ablation study.
API keys and model definitions for all providers.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
GENAI_DIR = BASE_DIR.parent
RESULTS_DIR = BASE_DIR / "results"
GRAPHS_DIR = BASE_DIR / "graphs"

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("LLM_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# Model Definitions
# ---------------------------------------------------------------------------
# Each model entry: (provider, model_id, display_name, base_url_or_None)
# Selected to be comparable to Groq's llama-3.3-70b-versatile (large, capable)

MODELS = {
    # --- Baseline (current production model) ---
    "groq_llama70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B (Groq)",
        "api_key": GROQ_API_KEY,
        "base_url": "https://api.groq.com/openai/v1",
    },

    # --- Google Gemini (free tier) ---
    "gemini_2_5_flash": {
        "provider": "google",
        "model_id": "models/gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash (Google)",
        "api_key": GOOGLE_API_KEY,
        "base_url": None,
    },

    # --- HuggingFace Inference API (free) ---
    "hf_qwen_72b": {
        "provider": "huggingface",
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "display_name": "Qwen 2.5 72B (HuggingFace)",
        "api_key": HUGGINGFACE_API_KEY,
        "base_url": None,
    },
}

# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
TEMPERATURE = 0.3
MAX_TOKENS = 2048
RETRY_ATTEMPTS = 2
RETRY_DELAY_SECONDS = 5
