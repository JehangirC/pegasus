"""Tests for the VertexAI LLM implementation."""

import unittest
from unittest.mock import MagicMock, patch
from typing import Dict, Any

from evaluator.llms.vertexai_llm import VertexAILLM


class TestVertexAILLM(unittest.TestCase):
    def setUp(self) -> None:
        self.default_config: Dict[str, str] = {
            "model_name": "gemini-1.0-pro",  # Use a valid model name
            "project_id": "test-project",
            "location": "us-central1",  # Use a valid location
        }

    @patch("evaluator.llms.vertexai_llm.vertexai.init")
    @patch("evaluator.llms.vertexai_llm.GenerativeModel")
    def test_initialization_success(self, mock_generative_model: MagicMock, mock_init: MagicMock) -> None:
        llm = VertexAILLM(**self.default_config)
        mock_init.assert_called_once_with(
            project=self.default_config["project_id"],
            location=self.default_config["location"],
        )
        self.assertEqual(llm.get_model_name(), self.default_config["model_name"])

    @patch("evaluator.llms.vertexai_llm.vertexai.init")
    @patch("evaluator.llms.vertexai_llm.GenerativeModel")
    def test_generate_success(self, mock_generative_model: MagicMock, mock_init: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_generative_model.return_value.generate_content.return_value = mock_response

        llm = VertexAILLM(**self.default_config)
        result = llm.generate("Test prompt")

        self.assertEqual(result, "This is a test response.")
        mock_generative_model.return_value.generate_content.assert_called_once_with(
            "Test prompt"
        )

    @patch("evaluator.llms.vertexai_llm.vertexai.init")
    @patch("evaluator.llms.vertexai_llm.GenerativeModel")
    def test_generate_with_parameters(self, mock_generative_model: MagicMock, mock_init: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_generative_model.return_value.generate_content.return_value = mock_response

        llm = VertexAILLM(**self.default_config)
        generation_params: Dict[str, Any] = {
            "temperature": 0.7,
            "max_output_tokens": 200,
            "top_p": 0.8,
            "top_k": 40,
        }

        result = llm.generate("Test prompt", **generation_params)

        self.assertEqual(result, "This is a test response.")
        mock_generative_model.return_value.generate_content.assert_called_once_with(
            "Test prompt", **generation_params
        )


if __name__ == "__main__":
    unittest.main()
