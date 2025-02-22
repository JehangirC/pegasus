"""Tests for the DeepEval evaluator implementation."""
import pytest
import pandas as pd
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.base_evaluator import EvaluationResult

class MockLLM:
    def generate(self, prompt: str) -> str:
        return "This is a mock response"
        
    def get_model_name(self) -> str:
        return "mock-model"

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
def deepeval_evaluator(mock_llm):
    return DeepEvalEvaluator(llm=mock_llm)

def test_deepeval_evaluator_initialization(mock_llm):
    evaluator = DeepEvalEvaluator(llm=mock_llm)
    assert evaluator.metrics == evaluator.default_metrics()
    
    custom_metrics = ["answer_relevancy", "faithfulness"]
    evaluator = DeepEvalEvaluator(metrics=custom_metrics, llm=mock_llm)
    assert evaluator.metrics == custom_metrics

def test_deepeval_evaluator_invalid_metric(mock_llm):
    with pytest.raises(ValueError):
        DeepEvalEvaluator(metrics=["invalid_metric"], llm=mock_llm)

def test_deepeval_evaluation(deepeval_evaluator, sample_df):
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

def test_deepeval_missing_llm():
    with pytest.raises(ValueError):
        DeepEvalEvaluator(llm=None)

def test_deepeval_supported_metrics(deepeval_evaluator):
    supported = deepeval_evaluator.supported_metrics()
    assert isinstance(supported, list)
    assert len(supported) > 0
    assert "answer_relevancy" in supported
    assert "faithfulness" in supported

def test_deepeval_list_context(deepeval_evaluator):
    df = pd.DataFrame({
        "question": ["Test question"],
        "answer": ["Test answer"],
        "context": [["Context 1", "Context 2"]],
        "expected_answer": ["Expected"]
    })
    results = deepeval_evaluator.evaluate(df)
    assert len(results) == 1
    assert len(results[0]) > 0