"""Evaluator implementation using Ragas metrics."""

import logging

import grpc

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import warnings
from typing import Dict, List

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

warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

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
        metrics: List[str] = None,
        threshold: float = None,
        llm=None,
        column_mapping: Dict[str, str] = None,
    ):
        super().__init__(metrics, threshold)
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

    def __del__(self):
        """Ensure proper cleanup of gRPC resources."""
        try:
            grpc.aio.shutdown_asyncio_engine()
        except:
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

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using Ragas metrics."""
        evaluation_results = {}

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
                    threshold = get_metric_threshold(metric_name, "ragas")
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric_name,
                            score=score,
                            passed=score >= threshold,
                            threshold=threshold,
                            explanation=f"Ragas {metric_name} score: {score:.3f}",
                            reason=f"Score {'meets' if score >= threshold else 'below'} threshold",
                        )
                    )
                evaluation_results[idx] = row_results

        except Exception as e:
            import traceback

            error_msg = f"Evaluation failed: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)  # Print the full error for debugging
            for idx in range(len(df)):
                evaluation_results[idx] = [
                    EvaluationResult(
                        metric_name=metric_name,
                        score=0.0,
                        passed=False,
                        threshold=get_metric_threshold(metric_name, "ragas"),
                        explanation=f"Error evaluating {metric_name}",
                        reason=error_msg,
                    )
                    for metric_name in self.metrics
                ]

        return evaluation_results

    def default_metrics(self) -> List[str]:
        """Returns default metrics for Ragas."""
        return DEFAULT_RAGAS_METRICS

    def supported_metrics(self) -> List[str]:
        """Returns all supported Ragas metrics."""
        return list(_SUPPORTED_METRICS.keys())
