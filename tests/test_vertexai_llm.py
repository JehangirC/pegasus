import unittest
from unittest.mock import patch, MagicMock  # Import MagicMock

from evaluator.llms.vertexai_llm import VertexAILLM


class TestVertexAILLM(unittest.TestCase):
    @patch("evaluator.llms.vertexai_llm.GenerativeModel")  # Mock GenerativeModel
    @patch("evaluator.llms.vertexai_llm.vertexai.init")  # Mock vertexai.init
    def test_generate_success(self, mock_init, mock_generative_model):
        # Mock the response from the Vertex AI model
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_generative_model.return_value.generate_content.return_value = (
            mock_response
        )

        # Create an instance of VertexAILLM (the mocks will be used)
        llm = VertexAILLM(
            model_name="test-model", project_id="test-project", location="test-location"
        )

        # Call the generate method
        result = llm.generate("Test prompt")

        # Assert that the result is correct
        self.assertEqual(result, "This is a test response.")

        # Assert that generate_content was called with the correct arguments
        mock_generative_model.return_value.generate_content.assert_called_once_with(
            "Test prompt"
        )
        mock_init.assert_called_once_with(
            project="test-project", location="test-location"
        )
        self.assertEqual(llm.get_model_name(), "test-model")


    @patch("evaluator.llms.vertexai_llm.GenerativeModel")
    @patch("evaluator.llms.vertexai_llm.vertexai.init")
    def test_generate_failure(self, mock_init, mock_generative_model):
        # Configure the mock to raise an exception
        mock_generative_model.return_value.generate_content.side_effect = Exception(
            "Test exception"
        )

        llm = VertexAILLM()

        # Assert that the exception is raised
        with self.assertRaises(Exception):
            llm.generate("Test prompt")

    @patch("evaluator.llms.vertexai_llm.GenerativeModel")  # Mock GenerativeModel
    @patch("evaluator.llms.vertexai_llm.vertexai.init")  # Mock vertexai.init
    def test_generate_success_kwargs(self, mock_init, mock_generative_model):
      # Mock the response from the Vertex AI model
        mock_response = MagicMock()
        mock_response.text = "This is a test response."
        mock_generative_model.return_value.generate_content.return_value = (
            mock_response
        )

        # Create an instance of VertexAILLM (the mocks will be used)
        llm = VertexAILLM(
            model_name="test-model", project_id="test-project", location="test-location", temperature=0.7
        )

        # Call the generate method
        result = llm.generate("Test prompt", temperature=0.2, max_output_tokens=200)

        # Assert that the result is correct
        self.assertEqual(result, "This is a test response.")

        # Assert that generate_content was called with the correct arguments
        mock_generative_model.return_value.generate_content.assert_called_once_with(
            "Test prompt", temperature=0.2, max_output_tokens=200
        )
        mock_init.assert_called_once_with(
            project="test-project", location="test-location"
        )
        self.assertEqual(llm.get_model_name(), "test-model")


if __name__ == "__main__":
    unittest.main()