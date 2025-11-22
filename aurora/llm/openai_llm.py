import tenacity
from openai import OpenAI
from .base import BaseLLM


class OpenAILLM(BaseLLM):
    """
    OpenAI model backend.
    Provides robust retry logic and unified chat interface.
    """

    def __init__(self,
                 name="OpenAILLM",
                 api_key="",
                 model="gpt-4o-mini",
                 temperature=0.2,
                 max_tokens=1000):
        
        super().__init__(name,
                         api_key=api_key,
                         model=model)
        
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _instantiate(self, api_key, model):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)

    @tenacity.retry(wait=tenacity.wait_exponential(min=1, max=30))
    def _completion(self, **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            **kwargs
        )

    def generate(self, messages):
        response = self._completion(
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens
        )
        return response.choices[0].message.content
