"""Tests for the Ragas evaluator implementation."""
import pytest
import pandas as pd
from evaluator.ragas_evaluator import RagasEvaluator
from evaluator.base_evaluator import EvaluationResult

class MockLLM:
    def invoke(self, prompt: str) -> str:
        return "This is a mock response"
    
    async def ainvoke(self, prompt: str) -> str:
        return "This is a mock async response"

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "question": ["What is the capital of France?", "What is 2+2?"],
        "answer": ["Paris is the capital of France.", "The answer is 4."],
        "context": ["France is a country in Europe.", "Basic arithmetic operations."],
        "expected_answer": ["Paris", "4"]
    })

@pytest.fixture
def mock_llm():
    return MockLLM()

@pytest.fixture
def ragas_evaluator():
    # Test default behavior without LLM
    return RagasEvaluator()

def test_ragas_evaluator_initialization():
    # Test without LLM (default config)
    evaluator = RagasEvaluator()
    assert evaluator.metrics == evaluator.default_metrics()
    
    # Test with custom metrics, no LLM
    custom_metrics = ["answer_relevancy", "faithfulness"]
    evaluator = RagasEvaluator(metrics=custom_metrics)
    assert evaluator.metrics == custom_metrics
    
    # Test with custom LLM
    mock_llm = MockLLM()
    evaluator = RagasEvaluator(llm=mock_llm)
    assert evaluator.metrics == evaluator.default_metrics()
    assert evaluator.llm is not None

def test_ragas_evaluator_invalid_metric():
    with pytest.raises(ValueError):
        RagasEvaluator(metrics=["invalid_metric"])

def test_ragas_evaluation(ragas_evaluator, sample_df):
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

def test_ragas_supported_metrics(ragas_evaluator):
    supported = ragas_evaluator.supported_metrics()
    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "answer_relevancy" in supported
    assert "faithfulness" in supported

def test_ragas_evaluator_single_metric():
    evaluator = RagasEvaluator(metrics="answer_relevancy")
    assert evaluator.metrics == ["answer_relevancy"]