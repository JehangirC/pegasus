"""Tests for the main LLMEvaluator interface."""
import pytest
import pandas as pd
from main import LLMEvaluator

@pytest.fixture
def sample_data():
    return {
        "question": ["What is the capital of France?", "What is 2+2?"],
        "answer": ["Paris is the capital of France.", "The answer is 4."],
        "context": ["France is a country in Europe.", "Basic arithmetic operations."],
        "expected_answer": ["Paris", "4"]
    }

@pytest.fixture
def ragas_evaluator():
    return LLMEvaluator(evaluator_type="ragas")

def test_evaluator_initialization():
    evaluator = LLMEvaluator(evaluator_type="ragas")
    assert evaluator.get_supported_metrics()
    assert evaluator.get_default_metrics()

def test_evaluator_with_dict(ragas_evaluator, sample_data):
    results = ragas_evaluator.evaluate(sample_data)
    assert len(results) == 2
    for idx in range(2):
        assert idx in results
        assert len(results[idx]) > 0
        for result in results[idx]:
            assert result.score >= 0 and result.score <= 1
            assert result.metric_name
            assert result.explanation

def test_evaluator_with_dataframe(ragas_evaluator, sample_data):
    df = pd.DataFrame(sample_data)
    results = ragas_evaluator.evaluate(df)
    assert len(results) == 2

def test_invalid_evaluator_type():
    with pytest.raises(ValueError):
        LLMEvaluator(evaluator_type="invalid")

def test_missing_required_columns(ragas_evaluator):
    data = {
        "question": ["What is the capital of France?"],
        # Missing "answer" column
    }
    with pytest.raises(ValueError):
        ragas_evaluator.evaluate(data)

def test_optional_columns(ragas_evaluator):
    data = {
        "question": ["What is the capital of France?"],
        "answer": ["Paris is the capital of France."]
    }
    # Should work without context and expected_answer
    results = ragas_evaluator.evaluate(data)
    assert len(results) == 1