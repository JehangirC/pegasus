"""Evaluator implementation using DeepEval metrics."""
from typing import Dict, List, Any
import pandas as pd
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    BiasMetric,
    ToxicityMetric
)
from deepeval import evaluate as deepeval_evaluate
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from langchain_google_vertexai import ChatVertexAI
from config import (
    PROJECT_ID, 
    LOCATION, 
    VERTEX_MODELS, 
    DEFAULT_DEEPEVAL_METRICS,
    get_metric_threshold
)

from .base_evaluator import BaseEvaluator, EvaluationResult

_SUPPORTED_METRICS = {
    "answer_relevancy": (AnswerRelevancyMetric, 0.5),
    "faithfulness": (FaithfulnessMetric, 0.5),
    "contextual_precision": (ContextualPrecisionMetric, 0.5),
    "contextual_recall": (ContextualRecallMetric, 0.5),
    "bias": (BiasMetric, 0.5),
    "toxicity": (ToxicityMetric, 0.5)
}

class GoogleVertexAIDeepEval(DeepEvalBaseLLM):
    """Wrapper for DeepEval to work with VertexAI"""
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

class DeepEvalEvaluator(BaseEvaluator):
    """Evaluator using the DeepEval library."""

    def __init__(self, metrics: List[str] = None, threshold: float = None, llm=None):
        super().__init__(metrics, threshold)
        self.validate_metrics(self.metrics)
        
        # Initialize VertexAI if no LLM provided
        if llm is None:
            vertex_llm = ChatVertexAI(
                model_name=VERTEX_MODELS["llm"]["name"],
                project=PROJECT_ID,
                location=LOCATION,
                generation_config=VERTEX_MODELS["llm"].get("config", {}),
                response_validation=False  # Important since deepeval cannot handle validation errors
            )
            self.llm = GoogleVertexAIDeepEval(model=vertex_llm)
        else:
            self.llm = llm

        # Initialize metrics
        self.deepeval_metrics = []
        for metric in self.metrics:
            MetricClass, _ = _SUPPORTED_METRICS[metric]
            threshold = get_metric_threshold(metric, "deepeval")
            self.deepeval_metrics.append(
                MetricClass(
                    threshold=threshold,
                    model=self.llm,
                    async_mode=False
                )
            )

    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates inputs using DeepEval metrics."""
        evaluation_results = {}
        
        for idx, row in df.iterrows():
            # Create test case for this row
            test_case = LLMTestCase(
                input=row["question"],
                actual_output=row["answer"],
                expected_output=row.get("expected_answer", ""),
                context=row["context"] if isinstance(row["context"], str) else "\n".join(row["context"])
            )
            
            # Evaluate with each metric
            row_results = []
            for metric in self.deepeval_metrics:
                try:
                    metric.measure(test_case)
                    score = float(metric.score)
                    metric_name = metric.__class__.__name__.replace("Metric", "").lower()
                    threshold = get_metric_threshold(metric_name, "deepeval")
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric_name,
                            score=score,
                            passed=score >= threshold,
                            threshold=threshold,
                            explanation=f"DeepEval {metric.__class__.__name__} score: {score:.3f}",
                            reason=metric.reason or f"Score {'meets' if score >= threshold else 'below'} threshold"
                        )
                    )
                except Exception as e:
                    metric_name = metric.__class__.__name__.replace("Metric", "").lower()
                    threshold = get_metric_threshold(metric_name, "deepeval")
                    row_results.append(
                        EvaluationResult(
                            metric_name=metric_name,
                            score=0.0,
                            passed=False,
                            threshold=threshold,
                            explanation=f"Error evaluating {metric.__class__.__name__}",
                            reason=str(e)
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