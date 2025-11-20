import os
from typing import List, Dict, Any

try:
    import openai
except ImportError as e:
    raise ImportError("Please install the 'openai' package to use GPT models.") from e


def _get_client() -> Any:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    openai.api_key = api_key
    return openai


def ask_gpt(prompt: str, model: str = "gpt-4.1-mini", max_tokens: int = 300) -> str:
    """
    Simple wrapper for a call to GPT
    """
    client = _get_client()
    response = client.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response["choices"][0]["message"]["content"]
