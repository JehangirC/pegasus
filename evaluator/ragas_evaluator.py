"""Evaluator implementation using Ragas metrics."""

import asyncio
import logging
import warnings
from typing import Any, Dict, List, Optional, Union

import grpc
import pandas as pd
from datasets import Dataset
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .base_evaluator import BaseEvaluator, EvaluationResult
from .config import (
    DEFAULT_RAGAS_METRICS,
    LOCATION,
    PROJECT_ID,
    VERTEX_MODELS,
    get_metric_threshold,
)

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

# Definition of metrics with their expected behavior
# Note: For all Ragas metrics, higher scores are better
_SUPPORTED_METRICS = {
    "answer_relevancy": answer_relevancy,
    "faithfulness": faithfulness,
    "context_recall": context_recall,
    "context_precision": context_precision,
    "answer_correctness": answer_correctness,
}


class RagasEvaluator(BaseEvaluator):
    """Evaluator using the Ragas library."""

    REQUIRED_COLUMNS = ["question", "answer", "context"]
    OPTIONAL_COLUMNS = ["ground_truth"]
    DEFAULT_COLUMN_MAPPING = {
        "question": "question",
        "answer": "answer",
        "context": "context",
        "ground_truth": "ground_truth",
    }

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        llm: Optional[Any] = None,
        column_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(metrics, threshold if threshold is not None else 0.5)
        self.validate_metrics(self.metrics)
        self.column_mapping = column_mapping or self.DEFAULT_COLUMN_MAPPING

        # Initialize gRPC channel with proper cleanup
        try:
            if llm is None:
                llm = VertexAI(
                    model_name=VERTEX_MODELS["llm"]["name"],
                    project=PROJECT_ID,
                    location=LOCATION,
                )

            self.llm = LangchainLLMWrapper(llm)

            # Initialize embeddings model
            self.embeddings = VertexAIEmbeddings(
                model_name=VERTEX_MODELS["embeddings"]["name"],
                project=PROJECT_ID,
                location=LOCATION,
            )
        except Exception as e:
            # Ensure proper cleanup if initialization fails
            grpc.aio.shutdown_asyncio_engine()
            raise e

    def __del__(self) -> None:
        """Ensure proper cleanup of gRPC resources."""
        try:
            grpc.aio.shutdown_asyncio_engine()
        except Exception as e:
            logging.debug(f"Error during gRPC shutdown: {e}")
            pass

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

    def _process_context(self, context: Union[str, List[str]]) -> List[str]:
        """Process context into expected format."""
        if isinstance(context, list):
            return context
        return [context]

    def _get_detailed_reason(
        self, metric_name: str, score: float, threshold: float
    ) -> str:
        """Get a detailed reason for the evaluation result."""
        base_reason = f"Score {score:.3f} {'meets' if score >= threshold else 'below'} threshold {threshold:.3f}"

        # Add metric-specific explanations
        if metric_name == "answer_relevancy":
            base_reason += "\nMeasures how relevant the answer is to the question."
        elif metric_name == "faithfulness":
            base_reason += (
                "\nMeasures if the answer is faithful to the provided context."
            )
        elif metric_name == "context_recall":
            base_reason += (
                "\nMeasures if all relevant information was retrieved from the context."
            )
        elif metric_name == "context_precision":
            base_reason += "\nMeasures if the retrieved context contains only relevant information."
        elif metric_name == "answer_correctness":
            base_reason += (
                "\nMeasures if the answer is correct according to the ground truth."
            )

        return base_reason

    def _create_evaluation_result(
        self,
        metric_name: str,
        score: float,
        threshold: float,
        error: Optional[Exception] = None,
    ) -> EvaluationResult:
        """Create an evaluation result for a metric."""
        if error:
            return EvaluationResult(
                metric_name=metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                explanation=f"Error evaluating {metric_name}: {type(error).__name__}",
                reason=f"Evaluation failed: {str(error)}",
            )

        return EvaluationResult(
            metric_name=metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            explanation=f"Ragas {metric_name} score: {score:.3f}",
            reason=self._get_detailed_reason(metric_name, score, threshold),
        )

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using Ragas metrics."""
        evaluation_results: Dict[str, List[EvaluationResult]] = {}

        # Map columns to expected names
        df_mapped = df.copy()

        # Validate required columns
        missing_cols = [
            col for col in self.REQUIRED_COLUMNS if col not in df_mapped.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Add retrieved_contexts column required by context_recall metric
        if "context" in df_mapped.columns:
            # Convert single context into list format required by Ragas
            df_mapped["retrieved_contexts"] = df_mapped["context"].apply(
                lambda x: [x] if isinstance(x, str) else x
            )

        # Convert DataFrame to Huggingface Dataset
        dataset = Dataset.from_pandas(df_mapped)

        # Get metric instances - instantiate them here
        metric_instances = [
            _SUPPORTED_METRICS[m] for m in self.metrics
        ]  # Note: instantiating metrics

        # Run evaluation
        try:
            results = evaluate(
                dataset,
                metrics=metric_instances,
                llm=self.llm,
                embeddings=self.embeddings,
            )

            # Process results for each row
            for idx in range(len(df)):
                row_results = []
                for metric_name in self.metrics:
                    score = float(results[metric_name][idx])
                    threshold = (
                        get_metric_threshold(metric_name, "ragas") or self.threshold
                    )
                    row_results.append(
                        self._create_evaluation_result(metric_name, score, threshold)
                    )
                evaluation_results[str(idx)] = row_results

        except Exception as e:
            import traceback

            error_msg = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
            logging.error(error_msg)  # Log the error instead of printing

            for idx in range(len(df)):
                row_results = []
                for metric_name in self.metrics:
                    threshold = (
                        get_metric_threshold(metric_name, "ragas") or self.threshold
                    )
                    row_results.append(
                        self._create_evaluation_result(
                            metric_name, 0.0, threshold, error=e
                        )
                    )
                evaluation_results[str(idx)] = row_results

        return evaluation_results

    async def evaluate_async(
        self, df: pd.DataFrame
    ) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using Ragas metrics asynchronously.

        Note: This is a placeholder implementation since Ragas doesn't support
        async evaluation directly. This method runs the synchronous evaluate
        method in an executor to avoid blocking the event loop.
        """
        # Use run_in_executor to run the synchronous evaluate method in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.evaluate, df)

    def default_metrics(self) -> List[str]:
        """Returns default metrics for Ragas."""
        return DEFAULT_RAGAS_METRICS

    def supported_metrics(self) -> List[str]:
        """Returns all supported Ragas metrics."""
        return list(_SUPPORTED_METRICS.keys())
