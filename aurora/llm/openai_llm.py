"""
OpenAI-backed LLM for AURORA.

Uses the OpenAI ChatCompletion API through python client.
"""

import os
from openai import OpenAI
from aurora.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    def __init__(self, model: str = "gpt-4.1-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set.")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message["content"]
