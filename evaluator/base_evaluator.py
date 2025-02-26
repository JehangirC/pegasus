"""Base evaluator class for LLM evaluation."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel, field_validator


class EvaluationResult(BaseModel):
    metric_name: str
    score: float
    explanation: str = ""
    reason: str = ""
    passed: bool = False
    threshold: float = 0.5


class EvaluationInput(BaseModel):
    question: str
    answer: str
    context: Union[str, List[str]]
    expected_answer: str = ""

    @field_validator("context")
    def context_must_be_valid(cls, v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, list):
            if not all(isinstance(item, str) for item in v):
                raise ValueError("All items in context must be strings if it is a list")
        elif not isinstance(v, str):
            raise ValueError("context must be a string or a list of strings")
        return v


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    def __init__(
        self, metrics: Optional[Union[str, List[str]]] = None, threshold: float = 0.5
    ) -> None:
        self.threshold = threshold
        # Convert single metric to list
        if isinstance(metrics, str):
            metrics = [metrics]
        self.metrics: List[str] = metrics or self.default_metrics()

    @abstractmethod
    def evaluate(self, df: pd.DataFrame) -> Dict[str, List[EvaluationResult]]:
        """Evaluates a dataframe of inputs and returns scores.

        Args:
            df: Pandas DataFrame with columns:
                - question: The input question
                - answer: The model's answer
                - context: The context provided (optional)
                - expected_answer: The expected answer (optional)

        Returns:
            Dictionary mapping each example to a list of evaluation results
        """
        pass

    @abstractmethod
    def default_metrics(self) -> List[str]:
        """Returns the default list of metrics for this evaluator."""
        pass

    @abstractmethod
    def supported_metrics(self) -> List[str]:
        """Returns all supported metrics for this evaluator."""
        pass

    def validate_metrics(self, metrics: List[str]) -> None:
        """Validates that the requested metrics are supported.

        Args:
            metrics: List of metric names to validate

        Raises:
            ValueError: If any metric is not supported
        """
        supported = self.supported_metrics()
        for metric in metrics:
            if metric not in supported:
                raise ValueError(
                    f"Metric '{metric}' not supported. Supported metrics: {supported}"
                )
