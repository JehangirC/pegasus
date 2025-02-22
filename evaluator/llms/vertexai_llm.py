import vertexai
from vertexai.generative_models import GenerativeModel
from .base_llm import BaseLLM
from typing import List, Dict, Any
import os
import logging
from deepeval.models import DeepEvalBaseLLM

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Default values, can be overridden by environment variables
DEFAULT_PROJECT_ID = os.getenv("PROJECT_ID")
DEFAULT_LOCATION = os.getenv("LOCATION")
DEFAULT_VERTEX_MODEL = os.getenv("DEFAULT_VERTEX_MODEL", "gemini-1.5-flash")  # Provide a default model


class VertexAILLM(BaseLLM):
    """Wrapper for Vertex AI Generative Models."""

    def __init__(self, model_name: str = DEFAULT_VERTEX_MODEL, project_id: str = DEFAULT_PROJECT_ID, location: str = DEFAULT_LOCATION, **kwargs):
        """
        Initializes the VertexAILLM.

        Args:
            model_name: The name of the Vertex AI model to use.
            project_id: Your Google Cloud project ID.
            location:  The Google Cloud location (e.g., "us-central1").
            **kwargs:  Additional keyword arguments to pass to the Vertex AI model
                (e.g., temperature, max_output_tokens, top_k, top_p).  These
                will override any defaults set during initialization.
        """
        super().__init__()
        self.use_vertex_ai = os.getenv("USE_VERTEX_AI", "True").lower() == "true"
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.kwargs = kwargs

        if self.use_vertex_ai:
            self._initialize_vertex_ai()
        else:
            logging.info("Running locally without Vertex AI.")
            self.model = None  # Or initialize a local model here

    def _initialize_vertex_ai(self):
        """Initializes the Vertex AI client."""
        try:
            vertexai.init(project=self.project_id, location=self.location)
            self.model = GenerativeModel(self.model_name, **self.kwargs)  # pass all kwargs to the model
            logging.info(f"Vertex AI initialized with model: {self.model_name}, project: {self.project_id}, location: {self.location}")
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise ValueError(f"Failed to initialize Vertex AI. Ensure your project ID and location are correct, and that you have the necessary permissions.  See the logs for details.") from e

    def generate(self, prompt: str, **kwargs) -> str:
        """Generates text from the Vertex AI model."""
        if not self.use_vertex_ai:
            logging.info("Generating text locally (no Vertex AI).")
            return "Local dummy response"

        try:
            # Combine instance kwargs with method kwargs
            all_kwargs = {**self.kwargs, **kwargs}
            response = self.model.generate_content(prompt, **all_kwargs)
            logging.info("Text generated successfully using Vertex AI.")
            return response.text  # return text directly
        except Exception as e:
            logging.error(f"Error during text generation with Vertex AI: {e}")  # Log exception, very important.
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