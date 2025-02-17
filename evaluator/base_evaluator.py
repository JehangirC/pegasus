from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pydantic import BaseModel, field_validator

class EvaluationResult(BaseModel):
    metric_name: str
    score: float
    #  Add other relevant fields, like explanations, supporting data, etc.

class EvaluationInput(BaseModel): # Pydantic classes for type checking input and output
    question: str
    answer: str
    context:  Union[str, List[str]] # can be a string or list of strings, depending on your Ragas setup.
    expected_answer: str = ""   # Optional:  For metrics that need a ground truth.

    @field_validator("context")
    def context_must_be_valid(cls, v):
        if isinstance(v, list):
            if not all(isinstance(item, str) for item in v):
                raise ValueError("All items in context must be strings if it is a list")
        elif not isinstance(v, str):
            raise ValueError("context must be a string or a list of strings")
        return v

class BaseEvaluator(ABC):
    """Abstract base class for all evaluators."""

    @abstractmethod
    def evaluate(self, inputs: List[EvaluationInput]) -> List[Dict[str,Any]]:
        """
        Evaluates a list of inputs and returns a list of scores.

        Args:
            inputs: A list of EvaluationInput objects.

        Returns:
            A list of dictionaries, where each dictionary contains
            the evaluation results for a single input. The exact structure
            will depend on the specific evaluator.
        """
        pass

    @abstractmethod
    def supported_metrics(self) -> List[str]:
      """return the list of metrics that can be computed by evaluator"""