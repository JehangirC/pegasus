"""Evaluator implementation using Ragas metrics."""
from typing import Dict, List, Any
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness
)
from datasets import Dataset
from ragas.llms.base import LangchainLLMWrapper
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from .base_evaluator import BaseEvaluator, EvaluationResult
from config import (
    PROJECT_ID, 
    LOCATION, 
    VERTEX_MODELS, 
    DEFAULT_RAGAS_METRICS,
    get_metric_threshold
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")

_SUPPORTED_METRICS = {
    "answer_relevancy": answer_relevancy,
    "faithfulness": faithfulness,
    "context_recall": context_recall,
    "context_precision": context_precision,
    "answer_correctness": answer_correctness
}

class RAGASVertexAIEmbeddings(VertexAIEmbeddings):
    """Wrapper for RAGAS to work with VertexAI embeddings"""
    async def embed_text(self, text: str) -> List[float]:
        """Embeds a text for semantics similarity"""
        return self.embed_documents([text])[0]

    def set_run_config(self, run_config):
        """Sets the run configuration for embeddings.
        
        Args:
            run_config: Configuration provided by RAGAS framework
        """
        pass  # VertexAI embeddings don't need runtime configuration

class RagasEvaluator(BaseEvaluator):
    """Evaluator using the Ragas library."""

    def __init__(self, metrics: List[str] = None, threshold: float = None, llm=None):
        super().__init__(metrics, threshold)
        self.validate_metrics(self.metrics)
        
        # Use provided LLM or create default from config
        if llm is not None:
            self.llm = LangchainLLMWrapper(llm)
        else:
            self.llm = LangchainLLMWrapper(VertexAI(
                model_name=VERTEX_MODELS["llm"]["name"],
                project=PROJECT_ID,
                location=LOCATION,
                **VERTEX_MODELS["llm"].get("config", {})
            ))

        # Initialize embeddings
        self.embeddings = RAGASVertexAIEmbeddings(
            model_name=VERTEX_MODELS["embeddings"]["name"],
            project=PROJECT_ID,
            location=LOCATION
        )
        
        # Initialize metrics with VertexAI
        self.ragas_metrics = []
        for metric_name in self.metrics:
            metric = _SUPPORTED_METRICS[metric_name]
            # Set LLM for the metric
            if hasattr(metric, "llm"):
                metric.__setattr__("llm", self.llm)
            # Set embeddings if the metric needs them
            if hasattr(metric, "embeddings"):
                metric.__setattr__("embeddings", self.embeddings)
            self.ragas_metrics.append(metric)

    def validate_metrics(self, metrics: List[str]) -> None:
        """Validates that provided metrics are supported."""
        if not metrics:
            self.metrics = self.default_metrics()
            return
            
        invalid_metrics = [m for m in metrics if m not in _SUPPORTED_METRICS]
        if invalid_metrics:
            raise ValueError(f"Unsupported metrics: {invalid_metrics}. Supported metrics are: {list(_SUPPORTED_METRICS.keys())}")

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using Ragas metrics."""
        try:
            # Convert pandas DataFrame to Ragas Dataset format
            dataset_dict = {
                "question": df["question"].tolist(),
                "answer": df["answer"].tolist(),
                "contexts": [[ctx] if isinstance(ctx, str) else ctx for ctx in df["context"].tolist()],
            }
            
            # Only add ground_truth if it exists
            if "expected_answer" in df.columns:
                dataset_dict["ground_truth"] = df["expected_answer"].tolist()
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run Ragas evaluation
            results = evaluate(dataset, metrics=self.ragas_metrics)
            results_df = results.to_pandas()

            # Convert results to EvaluationResult format
            evaluation_results = {}
            for idx in range(len(df)):
                row_results = []
                for metric in self.metrics:
                    if metric not in results_df.columns:
                        # Handle case where metric wasn't computed
                        threshold = get_metric_threshold(metric, "ragas")
                        row_results.append(
                            EvaluationResult(
                                metric_name=metric,
                                score=0.0,
                                passed=False,
                                threshold=threshold,
                                explanation=f"Ragas {metric} score not computed",
                                reason=f"Metric may require ground truth that wasn't provided"
                            )
                        )
                    else:
                        score = float(results_df.iloc[idx][metric])
                        threshold = get_metric_threshold(metric, "ragas")
                        row_results.append(
                            EvaluationResult(
                                metric_name=metric,
                                score=score,
                                passed=score >= threshold,
                                threshold=threshold,
                                explanation=f"Ragas {metric} score: {score:.3f}",
                                reason=f"Score {'meets' if score >= threshold else 'below'} threshold of {threshold}"
                            )
                        )
                evaluation_results[idx] = row_results
                
            return evaluation_results
        except Exception as e:
            # Handle any evaluation errors gracefully
            error_results = {}
            for idx in range(len(df)):
                row_results = []
                for metric in self.metrics:
                    threshold = get_metric_threshold(metric, "ragas")
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric,
                            score=0.0,
                            passed=False,
                            threshold=threshold,
                            explanation=f"Error evaluating with Ragas",
                            reason=str(e)
                        )
                    )
                error_results[idx] = row_results
            return error_results

    def default_metrics(self) -> List[str]:
        """Returns default metrics for Ragas evaluation."""
        return DEFAULT_RAGAS_METRICS

    def supported_metrics(self) -> List[str]:
        """Returns all supported Ragas metrics."""
        return list(_SUPPORTED_METRICS.keys())