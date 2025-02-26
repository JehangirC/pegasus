"""Tests for the DeepEval evaluator implementation."""

from typing import Any

import pandas as pd
import pytest

from evaluator.base_evaluator import EvaluationResult
from evaluator.deepeval_evaluator import DeepEvalEvaluator


class MockLLM:
    def __init__(self) -> None:
        self.model_name = "mock-model"

    def generate(self, prompt: str) -> str:
        return "This is a mock response"

    def get_model_name(self) -> str:
        return self.model_name

    def invoke(self, prompt: str) -> str:
        return self.generate(prompt)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": ["What is the capital of France?", "What is 2+2?"],
            "answer": ["Paris is the capital of France.", "The answer is 4."],
            "context": [
                "France is a country in Europe.",
                "Basic arithmetic operations.",
            ],
            "expected_answer": ["Paris", "4"],
        }
    )


@pytest.fixture
def mock_llm() -> MockLLM:
    return MockLLM()


@pytest.fixture
def deepeval_evaluator(mock_llm: MockLLM) -> DeepEvalEvaluator:
    return DeepEvalEvaluator(llm=mock_llm)


def test_initialization_with_defaults(mock_llm: MockLLM) -> None:
    evaluator = DeepEvalEvaluator(llm=mock_llm)
    assert evaluator.metrics == evaluator.default_metrics()
    assert evaluator.llm is not None


def test_initialization_with_custom_metrics(mock_llm: MockLLM) -> None:
    custom_metrics = ["answer_relevancy", "faithfulness"]
    evaluator = DeepEvalEvaluator(metrics=custom_metrics, llm=mock_llm)
    assert evaluator.metrics == custom_metrics
    assert evaluator.llm is not None

    with pytest.raises(ValueError, match="Invalid metrics"):
        DeepEvalEvaluator(metrics=["invalid_metric"], llm=MockLLM())


def test_single_metric_initialization(mock_llm: MockLLM) -> None:
    evaluator = DeepEvalEvaluator(metrics=["answer_relevancy"], llm=mock_llm)
    assert evaluator.metrics == ["answer_relevancy"]


def test_supported_metrics(deepeval_evaluator: DeepEvalEvaluator) -> None:
    supported = deepeval_evaluator.supported_metrics()
    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "answer_relevancy" in supported
    assert "faithfulness" in supported


def test_evaluation_basic(deepeval_evaluator: DeepEvalEvaluator, sample_df: pd.DataFrame) -> None:
    results = deepeval_evaluator.evaluate(sample_df)

    assert len(results) == len(sample_df)
    for idx in results:
        assert isinstance(results[idx], list)
        for result in results[idx]:
            assert isinstance(result, EvaluationResult)
            assert 0 <= result.score <= 1
            assert result.metric_name in deepeval_evaluator.supported_metrics()
            assert isinstance(result.passed, bool)
            assert result.explanation
            assert result.threshold > 0


def test_batch_evaluation(deepeval_evaluator: DeepEvalEvaluator) -> None:
    large_df = pd.DataFrame(
        {
            "question": ["Q" + str(i) for i in range(10)],
            "answer": ["A" + str(i) for i in range(10)],
            "context": ["C" + str(i) for i in range(10)],
            "expected_answer": ["E" + str(i) for i in range(10)],
        }
    )

    results = deepeval_evaluator.evaluate(large_df)
    assert len(results) == len(large_df)


def test_list_context_handling(deepeval_evaluator: DeepEvalEvaluator) -> None:
    df = pd.DataFrame(
        {
            "question": ["What is Python?"],
            "answer": ["Python is a programming language."],
            "context": [["Python is high-level.", "Python is interpreted."]],
            "expected_answer": ["A programming language"],
        }
    )

    results = deepeval_evaluator.evaluate(df)
    assert len(results) == 1
    assert all(isinstance(result, EvaluationResult) for result in results[0]) # type: ignore[index]


def test_evaluator_with_invalid_metric(mock_vertex: Any) -> None:
    """Test evaluator with invalid metric."""
    with pytest.raises(ValueError):
        DeepEvalEvaluator(metrics=["invalid_metric"])

    with pytest.raises(ValueError):
        DeepEvalEvaluator(metrics=["answer_relevancy", "invalid_metric"])


def test_evaluate(mock_vertex: Any, sample_input: pd.DataFrame) -> None:
    """Test evaluate method."""
    evaluator = DeepEvalEvaluator()
    results = evaluator.evaluate(sample_input)
    
    first_key = "0"
    # No type ignore needed since we're using string key
    assert len(results[first_key]) > 0
    assert all(isinstance(r, EvaluationResult) for r in results[first_key])
    assert all(hasattr(r, "score") for r in results[first_key])
