"""
Unified model client wrappers for all LLM providers.
Each client exposes a common interface: generate(system_prompt, user_prompt) -> str
"""

import time
import json
import requests
from openai import OpenAI

from comparative_analysis.config import TEMPERATURE, MAX_TOKENS, RETRY_ATTEMPTS, RETRY_DELAY_SECONDS


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class BaseLLMClient:
    """Common interface for all LLM providers."""

    def __init__(self, model_id: str, display_name: str, api_key: str):
        self.model_id = model_id
        self.display_name = display_name
        self.api_key = api_key

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Returns dict with:
          - response: str (raw text)
          - latency_seconds: float
          - input_tokens: int or None
          - output_tokens: int or None
          - error: str or None
        """
        raise NotImplementedError

    def health_check(self) -> dict:
        """Returns {"ok": bool, "message": str}."""
        raise NotImplementedError

    def _retry_generate(self, fn):
        """Retry wrapper with exponential backoff."""
        last_err = None
        for attempt in range(RETRY_ATTEMPTS + 1):
            try:
                return fn()
            except Exception as e:
                last_err = str(e)
                if attempt < RETRY_ATTEMPTS:
                    wait = RETRY_DELAY_SECONDS * (2 ** attempt)
                    print(f"  [retry {attempt+1}/{RETRY_ATTEMPTS}] {self.display_name} – waiting {wait}s …")
                    time.sleep(wait)
        return {
            "response": None,
            "latency_seconds": 0.0,
            "input_tokens": None,
            "output_tokens": None,
            "error": last_err,
        }


# ---------------------------------------------------------------------------
# Groq / OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------
class OpenAICompatibleClient(BaseLLMClient):
    """Works with Groq, OpenRouter, and any OpenAI-compatible endpoint."""

    def __init__(self, model_id: str, display_name: str, api_key: str, base_url: str, extra_headers: dict = None):
        super().__init__(model_id, display_name, api_key)
        self.base_url = base_url
        self.extra_headers = extra_headers or {}
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=self.extra_headers,
        )

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        def _call():
            t0 = time.time()
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            latency = time.time() - t0
            usage = resp.usage
            return {
                "response": resp.choices[0].message.content,
                "latency_seconds": round(latency, 3),
                "input_tokens": usage.prompt_tokens if usage else None,
                "output_tokens": usage.completion_tokens if usage else None,
                "error": None,
            }
        return self._retry_generate(_call)

    def health_check(self) -> dict:
        try:
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            return {"ok": True, "message": resp.choices[0].message.content.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Google Gemini (via google-generativeai SDK)
# ---------------------------------------------------------------------------
class GeminiClient(BaseLLMClient):
    """Google Gemini via the official google-generativeai SDK."""

    def __init__(self, model_id: str, display_name: str, api_key: str):
        super().__init__(model_id, display_name, api_key)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.genai = genai
            self.model = genai.GenerativeModel(
                model_name=model_id,
                system_instruction=None,  # set per-call
            )
        except ImportError:
            raise ImportError("Install google-generativeai: pip install google-generativeai")

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        def _call():
            import google.generativeai as genai
            model = genai.GenerativeModel(
                model_name=self.model_id,
                system_instruction=system_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                ),
            )
            t0 = time.time()
            resp = model.generate_content(user_prompt)
            latency = time.time() - t0

            input_tokens = None
            output_tokens = None
            if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                input_tokens = getattr(resp.usage_metadata, "prompt_token_count", None)
                output_tokens = getattr(resp.usage_metadata, "candidates_token_count", None)

            return {
                "response": resp.text,
                "latency_seconds": round(latency, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "error": None,
            }
        return self._retry_generate(_call)

    def health_check(self) -> dict:
        try:
            import google.generativeai as genai
            model = genai.GenerativeModel(model_name=self.model_id)
            resp = model.generate_content("Say OK")
            return {"ok": True, "message": resp.text.strip()}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# HuggingFace Inference API
# ---------------------------------------------------------------------------
class HuggingFaceClient(BaseLLMClient):
    """HuggingFace Inference API (serverless, free tier) via the new router."""

    HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

    def __init__(self, model_id: str, display_name: str, api_key: str):
        super().__init__(model_id, display_name, api_key)
        self.url = self.HF_API_URL
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, system_prompt: str, user_prompt: str) -> dict:
        def _call():
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }
            t0 = time.time()
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=120)
            latency = time.time() - t0

            if resp.status_code != 200:
                raise Exception(f"HF API error {resp.status_code}: {resp.text[:300]}")

            data = resp.json()

            # Parse chat completion response
            text = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})

            return {
                "response": text,
                "latency_seconds": round(latency, 3),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "error": None,
            }
        return self._retry_generate(_call)

    def health_check(self) -> dict:
        try:
            payload = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": "Say OK"}],
                "max_tokens": 10,
                "stream": False,
            }
            resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                return {"ok": True, "message": text.strip()}
            elif resp.status_code == 503:
                # Model loading – still consider the key valid
                return {"ok": True, "message": "Model loading (503) – key is valid"}
            else:
                return {"ok": False, "message": f"HTTP {resp.status_code}: {resp.text[:200]}"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_client(model_config: dict) -> BaseLLMClient:
    """Create the appropriate client from a model config dict."""
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    display_name = model_config["display_name"]
    api_key = model_config["api_key"]

    if provider == "groq":
        return OpenAICompatibleClient(
            model_id=model_id,
            display_name=display_name,
            api_key=api_key,
            base_url=model_config["base_url"],
        )
    elif provider == "openrouter":
        return OpenAICompatibleClient(
            model_id=model_id,
            display_name=display_name,
            api_key=api_key,
            base_url=model_config["base_url"],
            extra_headers={"HTTP-Referer": "https://lumin-ai.app", "X-Title": "LUMIN.AI Ablation Study"},
        )
    elif provider == "google":
        return GeminiClient(
            model_id=model_id,
            display_name=display_name,
            api_key=api_key,
        )
    elif provider == "huggingface":
        return HuggingFaceClient(
            model_id=model_id,
            display_name=display_name,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
