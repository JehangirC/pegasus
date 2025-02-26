"""Evaluator implementation using DeepEval metrics."""

import warnings
from typing import Dict, List

import pandas as pd
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    ToxicityMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_google_vertexai import ChatVertexAI

from .config import (
    DEFAULT_DEEPEVAL_METRICS,
    LOCATION,
    PROJECT_ID,
    VERTEX_MODELS,
    get_metric_threshold,
)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

from .base_evaluator import BaseEvaluator, EvaluationResult

_SUPPORTED_METRICS = {
    "answer_relevancy": (AnswerRelevancyMetric, 0.5),
    "faithfulness": (FaithfulnessMetric, 0.5),
    "contextual_precision": (ContextualPrecisionMetric, 0.5),
    "contextual_recall": (ContextualRecallMetric, 0.5),
    "bias": (BiasMetric, 0.5),
    "toxicity": (ToxicityMetric, 0.5),
}


class GoogleVertexAIDeepEval(DeepEvalBaseLLM):
    """Adapter for using Google Vertex AI with DeepEval."""

    def __init__(self, model: ChatVertexAI):
        self.model = model

    def load_model(self):
        return self.model

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def generate(self, prompt: str) -> str:
        """Generate response from the model."""
        response = self.model.invoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        """Get name of the model."""
        return f"Vertex AI - {self.model.model_name}"


class DeepEvalEvaluator(BaseEvaluator):
    """Evaluator using the DeepEval library."""

    def __init__(self, metrics: List[str] = None, threshold: float = None, llm=None):
        super().__init__(metrics, threshold)
        self.validate_metrics(self.metrics)

        if llm is None:
            llm = ChatVertexAI(
                model_name=VERTEX_MODELS["llm"]["name"],
                project=PROJECT_ID,
                location=LOCATION,
                verbose=False,
            )

        self.llm = GoogleVertexAIDeepEval(llm)

        # Initialize metrics
        self.deepeval_metrics = []
        for metric in self.metrics:
            if metric not in _SUPPORTED_METRICS:
                continue
            MetricClass, default_threshold = _SUPPORTED_METRICS[metric]
            threshold = get_metric_threshold(metric, "deepeval") or default_threshold
            self.deepeval_metrics.append(
                MetricClass(threshold=threshold, model=self.llm, async_mode=False)
            )

    def validate_metrics(self, metrics: List[str]) -> None:
        """Validates that provided metrics are supported."""
        if not metrics:
            self.metrics = self.default_metrics()
            return

        invalid_metrics = [m for m in metrics if m not in _SUPPORTED_METRICS]
        if invalid_metrics:
            raise ValueError(
                f"Unsupported metrics: {invalid_metrics}. Supported metrics are: {list(_SUPPORTED_METRICS.keys())}"
            )

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using DeepEval metrics."""
        evaluation_results = {}

        # Create mapping of metric class names to supported metric names
        metric_name_map = {
            cls.__name__: name for name, (cls, _) in _SUPPORTED_METRICS.items()
        }

        for idx, row in df.iterrows():
            context = (
                row["context"] if isinstance(row["context"], list) else [row["context"]]
            )
            test_case = LLMTestCase(
                input=row["question"],
                actual_output=row["answer"],
                expected_output=row.get("expected_answer", ""),
                context=context,
                retrieval_context=context,
            )

            # Evaluate with each metric
            row_results = []
            for metric in self.deepeval_metrics:
                try:
                    metric.measure(test_case)
                    score = float(metric.score)
                    metric_name = metric_name_map[metric.__class__.__name__]
                    threshold = (
                        get_metric_threshold(metric_name, "deepeval")
                        or _SUPPORTED_METRICS[metric_name][1]
                    )
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric_name,
                            score=score,
                            passed=score >= threshold,
                            threshold=threshold,
                            explanation=f"DeepEval {metric.__class__.__name__} score: {score:.3f}",
                            reason=metric.reason
                            or f"Score {'meets' if score >= threshold else 'below'} threshold",
                        )
                    )
                except Exception as e:
                    metric_name = metric_name_map[metric.__class__.__name__]
                    threshold = (
                        get_metric_threshold(metric_name, "deepeval")
                        or _SUPPORTED_METRICS[metric_name][1]
                    )
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric_name,
                            score=0.0,
                            passed=False,
                            threshold=threshold,
                            explanation=f"Error evaluating {metric.__class__.__name__}",
                            reason=str(e),
                        )
                    )

            evaluation_results[idx] = row_results

        return evaluation_results

    def default_metrics(self) -> List[str]:
        """Returns default metrics for DeepEval."""
        return DEFAULT_DEEPEVAL_METRICS

    def supported_metrics(self) -> List[str]:
        """Returns all supported DeepEval metrics."""
        return list(_SUPPORTED_METRICS.keys())
