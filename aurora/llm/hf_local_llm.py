"""
HuggingFace local backend for AURORA.

Supports any local model that works with 'text-generation' pipeline:
 - meta-llama/Llama-3-8B-Instruct
 - mistralai/Mistral-7B-Instruct
 - Qwen2.5 7B
"""

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from aurora.llm.base_llm import BaseLLM


class HFLocalLLM(BaseLLM):
    def __init__(self, model_name: str = "meta-llama/Llama-3-8B-Instruct",
                 device: str = "cpu"):
        self.model_name = model_name
        self.device = device

        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device,
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=device,
            max_new_tokens=512,
        )

    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        out = self.pipe(prompt, max_new_tokens=max_tokens)
        return out[0]["generated_text"]
