"""Tests for the Ragas evaluator implementation."""


import pandas as pd
import pytest

from evaluator.base_evaluator import EvaluationResult
from evaluator.ragas_evaluator import RagasEvaluator


class MockLLM:
    def __init__(self) -> None:
        self.model_name = "mock-model"

    def invoke(self, prompt: str) -> str:
        return "This is a mock response"

    async def ainvoke(self, prompt: str) -> str:
        return "This is a mock async response"

    def get_model_name(self) -> str:
        return self.model_name


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
def ragas_evaluator() -> RagasEvaluator:
    return RagasEvaluator(llm=None)  # Explicitly set llm to None for testing


def test_initialization_with_defaults() -> None:
    evaluator = RagasEvaluator()  # Let it use default llm
    assert evaluator.metrics == evaluator.default_metrics()


def test_initialization_with_custom_metrics() -> None:
    custom_metrics = ["answer_relevancy", "context_recall"]
    evaluator = RagasEvaluator(metrics=custom_metrics, llm=None)
    assert evaluator.metrics == custom_metrics


def test_initialization_with_llm(mock_llm: MockLLM) -> None:
    evaluator = RagasEvaluator(llm=mock_llm)
    assert evaluator.llm is not None
    assert evaluator.metrics == evaluator.default_metrics()


def test_initialization_validation() -> None:
    with pytest.raises(ValueError, match="Unsupported metrics"):
        RagasEvaluator(metrics=["invalid_metric"], llm=None)


def test_single_metric_initialization() -> None:
    evaluator = RagasEvaluator(metrics=["answer_relevancy"], llm=None)
    assert evaluator.metrics == ["answer_relevancy"]


def test_supported_metrics(ragas_evaluator: RagasEvaluator) -> None:
    supported = ragas_evaluator.supported_metrics()
    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "answer_relevancy" in supported
    assert "context_recall" in supported


def test_evaluation_basic(
    ragas_evaluator: RagasEvaluator, sample_df: pd.DataFrame
) -> None:
    results = ragas_evaluator.evaluate(sample_df)

    assert len(results) == len(sample_df)
    for idx in results:
        assert isinstance(results[idx], list)
        for result in results[idx]:
            assert isinstance(result, EvaluationResult)
            assert 0 <= result.score <= 1
            assert result.metric_name in ragas_evaluator.supported_metrics()
            assert isinstance(result.passed, bool)
            assert result.explanation
            assert result.threshold > 0


def test_evaluation_with_missing_context() -> None:
    evaluator = RagasEvaluator(metrics=["answer_relevancy"], llm=None)
    df = pd.DataFrame(
        {
            "question": ["What is Python?"],
            "answer": ["Python is a programming language."],
            "expected_answer": ["A programming language"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluator.evaluate(df)


def test_batch_evaluation(ragas_evaluator: RagasEvaluator) -> None:
    large_df = pd.DataFrame(
        {
            "question": ["Q" + str(i) for i in range(10)],
            "answer": ["A" + str(i) for i in range(10)],
            "context": ["C" + str(i) for i in range(10)],
            "expected_answer": ["E" + str(i) for i in range(10)],
        }
    )

    results = ragas_evaluator.evaluate(large_df)
    assert len(results) == len(large_df)


def test_metric_dependencies() -> None:
    evaluator = RagasEvaluator(metrics=["context_recall"], llm=None)
    df = pd.DataFrame(
        {
            "question": ["What is Python?"],
            "answer": ["Python is a programming language."],
            "expected_answer": ["A programming language"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        evaluator.evaluate(df)
