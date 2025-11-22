from abc import ABC, abstractmethod

class BaseLLM(ABC):
    """
    Abstract interface for all LLM backends used in AURORA.
    All agents must call models exclusively via generate(messages).
    """

    def __init__(self, name: str, **kwargs):
        self.name = name
        self._instantiate(**kwargs)

    @abstractmethod
    def _instantiate(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def generate(self, messages: list[dict]) -> str:
        raise NotImplementedError
