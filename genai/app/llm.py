"""Thin wrapper around the OpenAI-compatible API (works with Groq, OpenAI, etc.)."""

from openai import OpenAI
from app.config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL


class LLMClient:
    def __init__(self):
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        self.model = LLM_MODEL

    # ---- single-turn ---------------------------------------------------
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise Exception("LLM API rate limit exceeded. Please wait or upgrade your plan.")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                raise Exception("LLM API authentication failed. Check your API key.")
            else:
                raise Exception(f"LLM API error: {error_msg}")

    # ---- multi-turn ----------------------------------------------------
    def generate_with_history(
        self,
        system_prompt: str,
        messages: list,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        try:
            all_msgs = [{"role": "system", "content": system_prompt}] + messages
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=all_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower() or "429" in error_msg:
                raise Exception("LLM API rate limit exceeded. Please wait or upgrade your plan.")
            elif "401" in error_msg or "authentication" in error_msg.lower():
                raise Exception("LLM API authentication failed. Check your API key.")
            else:
                raise Exception(f"LLM API error: {error_msg}")

    # ---- health check --------------------------------------------------
    def check_connection(self) -> bool:
        try:
            self.client.models.list()
            return True
        except Exception:
            return False
