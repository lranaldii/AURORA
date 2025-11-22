"""
Utility function to construct the LLM backend from configuration.

Usage:
    llm = get_llm_backend("openai", model="gpt-4.1-mini")
    llm = get_llm_backend("hf", model="meta-llama/Llama-3-8B-Instruct")
"""

from aurora.llm.openai_llm import OpenAILLM
from aurora.llm.hf_local_llm import HFLocalLLM


def get_llm_backend(provider: str, model: str, device: str = "cpu"):
    provider = provider.lower()

    if provider == "openai":
        return OpenAILLM(model=model)

    elif provider in {"hf", "huggingface"}:
        return HFLocalLLM(model_name=model, device=device)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
