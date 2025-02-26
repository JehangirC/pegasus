"""Vertex AI LLM implementation."""

from typing import Any, Dict, List, Optional
import google.auth
import google.auth.transport.requests
from langchain_google_vertexai import ChatVertexAI
from ..config import PROJECT_ID, LOCATION, VERTEX_MODELS
from .base_llm import BaseLLM
import vertexai
from vertexai.generative_models import GenerativeModel
import logging
from deepeval.models import DeepEvalBaseLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class VertexAILLM(BaseLLM):
    """Wrapper for Vertex AI Generative Models."""

    def __init__(
        self,
        model_name: str = None,
        project_id: str = None,
        location: str = None,
        **kwargs,
    ):
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
        self.kwargs = kwargs
        self._initialize_vertex_ai()

    def _initialize_vertex_ai(self):
        """Initializes the Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(
                self.model_name, **self.kwargs
            )  # pass all kwargs to the model
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise ValueError(
                f"Failed to initialize Vertex AI. Ensure your project ID and location are correct, and that you have the necessary permissions.  See the logs for details."
            ) from e

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates text from the Vertex AI model."""
        try:
            # Combine instance kwargs with method kwargs
            all_kwargs = {**self.kwargs, **kwargs}
            response = self.model.generate_content(prompt, **all_kwargs)
            logging.info("Text generated successfully using Vertex AI.")
            return response.text
        except Exception as e:
            logging.error(
                f"Error during text generation with Vertex AI: {e}"
            )  # Log exception, very important.
            raise  # Re-raise the exception to stop execution.

    def get_model_name(self) -> str:
        return self.model_name

    def set_run_config(self, run_config):
        """
        Sets the run configuration for the LLM.

        Args:
            run_config: The run configuration to set.
        """
        self.run_config = run_config
        # You might need to do something with the run_config here,
        # depending on how ragas uses it.  For example, you might
        # want to store it as an instance variable.
        logging.info("Run config set")


class VertexAIDeepEvalWrapper(DeepEvalBaseLLM):
    def __init__(self, vertex_ai_llm):
        self.vertex_ai_llm = vertex_ai_llm

    def load_model(self):
        return self.vertex_ai_llm

    def generate(self, prompt: str) -> str:
        return self.vertex_ai_llm.generate(prompt)

    async def a_generate(self, prompt: str) -> str:
        return await self.vertex_ai_llm.a_generate(prompt)

    def get_model_name(self):
        return self.vertex_ai_llm.get_model_name()
