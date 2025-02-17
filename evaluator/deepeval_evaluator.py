from typing import List, Dict, Any
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    BiasMetric,
    ToxicityMetric
)  # Import specific metrics
from deepeval import evaluate as deepeval_evaluate, assert_test
from deepeval.test_case import LLMTestCase

from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult


_SUPPORTED_DEEPEVAL_METRICS = { # Keep track of which metrics and minimum score
    "answer_relevancy": (AnswerRelevancyMetric,0.7),
    "faithfulness": (FaithfulnessMetric, 0.7),
    "contextual_precision": (ContextualPrecisionMetric,0.7),
    "contextual_recall": (ContextualRecallMetric,0.7),
    "bias": (BiasMetric, 0.7),
    "toxicity": (ToxicityMetric,0.7),
}

class DeepEvalEvaluator(BaseEvaluator):
    """Evaluator using the DeepEval library."""

    def __init__(self, metrics: List[str] = None, model:str = "gpt-4"):
        """
        Initializes the DeepEvalEvaluator.

        Args:
            metrics: A list of metric names to use. If None, uses a default set.
            model: Name of the LLM to use for evaluation.
        """
        super().__init__()
        self.model = model #important for DeepEval
        if metrics is None:
            metrics = ["answer_relevancy", "faithfulness"]  # Default metrics

        self.metrics = self._validate_metrics(metrics)

    def _validate_metrics(self, metrics: List[str]):
        """Validates and returns the DeepEval metric objects."""
        validated_metrics = []
        for metric_name in metrics:
            if metric_name not in _SUPPORTED_DEEPEVAL_METRICS:
                raise ValueError(
                    f"Invalid DeepEval metric: {metric_name}.  "
                    f"Supported metrics: {list(_SUPPORTED_DEEPEVAL_METRICS.keys())}"
                )
            MetricClass, min_score = _SUPPORTED_DEEPEVAL_METRICS[metric_name]
            validated_metrics.append(MetricClass(minimum_score=min_score))  # Use min_score
        return validated_metrics



    def evaluate(self, inputs: List[EvaluationInput]) -> List[Dict[str, Any]]:
        """Evaluates a list of inputs using DeepEval."""
        results_list = []

        for inp in inputs:
            # Create a DeepEval TestCase for each input.
            test_case = LLMTestCase(
                input=inp.question,
                actual_output=inp.answer,
                context=inp.context,
                expected_output = inp.expected_answer
            )
            # You could use deepeval_evaluate for aggregated metrics
            # but assert_test gives a pass/fail per metric, per test case.
            # Here's how to use assert_test (more typical for DeepEval):

            metric_results: Dict[str, Any] = {}
            for metric in self.metrics:
              assert_test(test_case, [metric]) #metric is already initialized with min_score
              metric_results[metric.__name__] = metric.score

            results_list.append(metric_results)

        return results_list


    def supported_metrics(self) -> List[str]:
        return list(_SUPPORTED_DEEPEVAL_METRICS.keys())