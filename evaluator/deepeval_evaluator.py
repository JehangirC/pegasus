"""Evaluator implementation using DeepEval metrics."""

import asyncio
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

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

from .base_evaluator import BaseEvaluator, EvaluationResult
from .config import (
    DEFAULT_DEEPEVAL_METRICS,
    LOCATION,
    PROJECT_ID,
    VERTEX_MODELS,
    get_metric_threshold,
)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

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

    def __init__(self, model: ChatVertexAI) -> None:
        self.model = model

    def load_model(self) -> ChatVertexAI:
        return self.model

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return str(res.content)

    def generate(self, prompt: str) -> str:
        """Generate response from the model."""
        response = self.model.invoke(prompt)
        return str(response.content)

    def get_model_name(self) -> str:
        """Get name of the model."""
        return f"Vertex AI - {self.model.model_name}"


class DeepEvalEvaluator(BaseEvaluator):
    """Evaluator using the DeepEval library."""

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        llm: Optional[Any] = None,
    ) -> None:
        super().__init__(metrics, threshold if threshold is not None else 0.5)
        self.validate_metrics(self.metrics)

        if llm is None:
            llm = ChatVertexAI(
                model_name=VERTEX_MODELS["llm"]["name"],
                project=PROJECT_ID,
                location=LOCATION,
                verbose=False,
            )

        self.llm = GoogleVertexAIDeepEval(llm)
        self.deepeval_metrics: List[Any] = []

        # Initialize metrics
        for metric in self.metrics:
            if metric not in _SUPPORTED_METRICS:
                continue
            MetricClass, default_threshold = _SUPPORTED_METRICS[metric]
            threshold = get_metric_threshold(metric, "deepeval") or default_threshold
            self.deepeval_metrics.append(
                MetricClass(threshold=threshold, model=self.llm, async_mode=False)
            )

    def validate_metrics(self, metrics: Optional[List[str]]) -> None:
        """Validates that provided metrics are supported."""
        if not metrics:
            self.metrics = self.default_metrics()
            return

        invalid_metrics = [m for m in metrics if m not in _SUPPORTED_METRICS]
        if invalid_metrics:
            raise ValueError(
                f"Unsupported metrics: {invalid_metrics}. Supported metrics are: {list(_SUPPORTED_METRICS.keys())}"
            )

    def _process_context(self, context: Union[str, List[str], pd.Series]) -> List[str]:
        """Process context into expected format."""
        if isinstance(context, list):
            return context
        elif isinstance(context, pd.Series):
            # Convert pandas Series to string or list depending on content
            if isinstance(context.iloc[0], str):
                return [str(context.iloc[0])]
            elif isinstance(context.iloc[0], list):
                return context.iloc[0]
            return [str(context)]
        return [context]

    def _create_test_case(self, row: pd.Series) -> LLMTestCase:
        """Create a DeepEval test case from a DataFrame row."""
        context = self._process_context(row["context"])
        return LLMTestCase(
            input=row["question"],
            actual_output=row["answer"],
            expected_output=row.get("expected_answer", ""),
            context=context,
            retrieval_context=context,
        )

    def _get_detailed_reason(self, metric: Any, score: float, threshold: float) -> str:
        """Get a detailed reason for the evaluation result."""
        base_reason = (
            metric.reason if hasattr(metric, "reason") and metric.reason else ""
        )
        if not base_reason:
            base_reason = f"Score {score:.3f} {'meets' if score >= threshold else 'below'} threshold {threshold:.3f}"

        # Add metric-specific details if available
        if hasattr(metric, "evaluation_details") and metric.evaluation_details:
            base_reason += f"\nDetails: {str(metric.evaluation_details)}"

        return base_reason

    def _evaluate_single_metric(
        self, metric: Any, test_case: LLMTestCase, metric_name: str, threshold: float
    ) -> EvaluationResult:
        """Evaluate a single metric and return result."""
        try:
            metric.measure(test_case)
            score = float(metric.score)

            # For bias and toxicity metrics, lower scores are better (below threshold is passing)
            # For other metrics, higher scores are better (above threshold is passing)
            if metric_name in ["bias", "toxicity"]:
                passed = score <= threshold
            else:
                passed = score >= threshold

            return EvaluationResult(
                metric_name=metric_name,
                score=score,
                passed=passed,
                threshold=threshold,
                explanation=f"DeepEval {metric.__class__.__name__} score: {score:.3f}",
                reason=self._get_detailed_reason(metric, score, threshold),
            )
        except Exception as e:
            # Provide more specific error handling
            error_type = type(e).__name__
            error_message = str(e)
            return EvaluationResult(
                metric_name=metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                explanation=f"Error evaluating {metric.__class__.__name__}: {error_type}",
                reason=f"Evaluation failed: {error_message}",
            )

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using DeepEval metrics."""
        evaluation_results: Dict[str, List[EvaluationResult]] = {}

        # Create mapping of metric class names to supported metric names
        metric_name_map = {
            cls.__name__: name for name, (cls, _) in _SUPPORTED_METRICS.items()
        }

        for idx, row in df.iterrows():
            test_case = self._create_test_case(row)

            # Evaluate with each metric
            row_results: List[EvaluationResult] = []
            for metric in self.deepeval_metrics:
                metric_name = metric_name_map[metric.__class__.__name__]
                threshold = (
                    get_metric_threshold(metric_name, "deepeval")
                    or _SUPPORTED_METRICS[metric_name][1]
                )
                result = self._evaluate_single_metric(
                    metric, test_case, metric_name, threshold
                )
                row_results.append(result)

            evaluation_results[str(idx)] = row_results

        return evaluation_results

    async def evaluate_async(
        self, df: pd.DataFrame
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using DeepEval metrics asynchronously."""
        evaluation_results: Dict[str, List[EvaluationResult]] = {}

        # Create mapping of metric class names to supported metric names
        metric_name_map = {
            cls.__name__: name for name, (cls, _) in _SUPPORTED_METRICS.items()
        }

        # Process each row
        async def process_row(
            idx: int, row: pd.Series
        ) -> Tuple[str, List[EvaluationResult]]:
            test_case = self._create_test_case(row)
            row_results: List[EvaluationResult] = []

            # Create tasks for each metric
            tasks = []
            for metric in self.deepeval_metrics:
                metric_name = metric_name_map[metric.__class__.__name__]
                threshold = (
                    get_metric_threshold(metric_name, "deepeval")
                    or _SUPPORTED_METRICS[metric_name][1]
                )
                # Use run_in_executor to run synchronous measure method in a thread pool
                tasks.append(
                    self._evaluate_single_metric(
                        metric, test_case, metric_name, threshold
                    )
                )

            # Gather results
            for result in tasks:
                row_results.append(result)

            return str(idx), row_results

        # Process rows concurrently
        tasks = [process_row(idx, row) for idx, row in df.iterrows()]  # type: ignore[arg-type]
        results = await asyncio.gather(*tasks)

        # Organize results
        for idx, row_results in results:
            evaluation_results[idx] = row_results

        return evaluation_results

    def default_metrics(self) -> List[str]:
        """Returns default metrics for DeepEval."""
        return DEFAULT_DEEPEVAL_METRICS

    def supported_metrics(self) -> List[str]:
        """Returns all supported DeepEval metrics."""
        return list(_SUPPORTED_METRICS.keys())
