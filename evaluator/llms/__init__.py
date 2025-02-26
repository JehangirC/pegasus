from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for all LLM wrappers."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """
        Generates text from the LLM based on the given prompt.

        Args:
            prompt: The input text prompt.
            **kwargs:  Additional keyword arguments that might be needed by
                specific LLM implementations (e.g., temperature, max_tokens).

        Returns:
            The generated text from the LLM.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """returns the model name"""
