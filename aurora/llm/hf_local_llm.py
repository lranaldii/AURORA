import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM


class HuggingFaceLLM(BaseLLM):
    """
    Local HuggingFace LLM backend.
    Handles GPU placement and chat template rendering.
    """

    def __init__(self,
                 name="HuggingFaceLLM",
                 model_id="meta-llama/Meta-Llama-3-8B-Instruct",
                 max_new_tokens=800,
                 do_sample=False):

        super().__init__(name, model_id=model_id)
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

    def _instantiate(self, model_id):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, messages):
        enc = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(
            enc,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            pad_token_id=self.tokenizer.eos_token_id
        )

        full = self.tokenizer.decode(output[0])
        prompt = self.tokenizer.decode(enc[0])

        return full[len(prompt):].strip()
