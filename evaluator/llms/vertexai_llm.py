"""Vertex AI LLM implementation."""

import logging
from typing import Any, Dict, Optional

import vertexai
from deepeval.models import DeepEvalBaseLLM
from vertexai.generative_models import GenerativeModel

from ..config import LOCATION, PROJECT_ID, VERTEX_MODELS
from .base_llm import BaseLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VertexAILLM(BaseLLM):
    """Wrapper for Vertex AI Generative Models."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the VertexAILLM.

        Args:
            model_name: The name of the Vertex AI model to use.
            project_id: Your Google Cloud project ID.
            location:  The Google Cloud location (e.g., "us-central1").
            **kwargs:  Additional keyword arguments to pass to the Vertex AI model.
        """
        super().__init__()
        self.project_id = project_id or PROJECT_ID
        self.location = location or LOCATION
        self.model_name = model_name or VERTEX_MODELS["llm"]["name"]
        self.kwargs: Dict[str, Any] = kwargs
        self._initialize_vertex_ai()

    def _initialize_vertex_ai(self) -> None:
        """Initializes the Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(
                self.model_name, **self.kwargs
            )  # pass all kwargs to the model
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise ValueError(
                "Failed to initialize Vertex AI. Ensure your project ID and location are correct, and that you have the necessary permissions. See the logs for details."
            ) from e

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generates text from the Vertex AI model."""
        try:
            # Combine instance kwargs with method kwargs
            all_kwargs = {**self.kwargs, **kwargs}
            response = self.model.generate_content(prompt, **all_kwargs)
            logging.info("Text generated successfully using Vertex AI.")
            return str(response.text)  # Explicitly convert to str
        except Exception as e:
            logging.error(f"Error during text generation with Vertex AI: {e}")
            raise

    async def a_generate(self, prompt: str) -> str:
        """Generates text from the Vertex AI model asynchronously."""
        try:
            # Currently Vertex AI doesn't have native async support, so we'll use sync method
            return self.generate(prompt)
        except Exception as e:
            logging.error(f"Error during async text generation with Vertex AI: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name

    def set_run_config(self, run_config: Dict[str, Any]) -> None:
        """
        Sets the run configuration for the LLM.

        Args:
            run_config: The run configuration to set.
        """
        self.run_config = run_config
        logging.info("Run config set")


class VertexAIDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self, vertex_ai_llm: VertexAILLM) -> None:
        self.vertex_ai_llm = vertex_ai_llm

    def load_model(self) -> VertexAILLM:
        return self.vertex_ai_llm

    def generate(self, prompt: str) -> str:
        return self.vertex_ai_llm.generate(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.vertex_ai_llm.a_generate(prompt)

    def get_model_name(self) -> str:
        return self.vertex_ai_llm.get_model_name()
