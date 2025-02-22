"""Main interface for LLM evaluation."""
import os
import google.auth
import google.auth.transport.requests
import pandas as pd
from typing import List, Dict, Any, Union
from evaluator.ragas_evaluator import RagasEvaluator
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.llms.vertexai_llm import VertexAILLM
from evaluator.base_evaluator import EvaluationInput, EvaluationResult
from evaluator.llms.base_llm import BaseLLM
from config import (
    PROJECT_ID,
    LOCATION,
    VERTEX_MODELS,
    DEFAULT_RAGAS_METRICS,
    DEFAULT_DEEPEVAL_METRICS,
)

class LLMEvaluator:
    """Main class for evaluating LLM outputs."""
    
    def __init__(self, evaluator_type: str = "ragas", metrics: List[str] = None, threshold: float = 0.5, llm=None):
        """Initialize the evaluator.
        
        Args:
            evaluator_type: Type of evaluator to use ("ragas" or "deepeval")
            metrics: List of metrics to evaluate. If None, uses default metrics
            threshold: Score threshold for passing evaluation
            llm: Language model to use for evaluation (required for DeepEval)
        """
        if evaluator_type == "ragas":
            self.evaluator = RagasEvaluator(metrics=metrics, threshold=threshold)
        elif evaluator_type == "deepeval":
            self.evaluator = DeepEvalEvaluator(metrics=metrics, threshold=threshold, llm=llm)
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

    def evaluate(self, data: Union[pd.DataFrame, Dict[str, List[str]]]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate LLM outputs.
        
        Args:
            data: DataFrame or dictionary with columns/keys:
                - question: Input questions
                - answer: Model answers
                - context: Context provided (optional)
                - expected_answer: Expected answers (optional)
                
        Returns:
            Dictionary mapping each example index to a list of evaluation results
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data
            
        # Validate required columns
        required_cols = ["question", "answer"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Add optional columns if missing
        if "context" not in df.columns:
            df["context"] = ""
        if "expected_answer" not in df.columns:
            df["expected_answer"] = ""
            
        return self.evaluator.evaluate(df)
    
    def get_supported_metrics(self) -> List[str]:
        """Get list of metrics supported by the current evaluator."""
        return self.evaluator.supported_metrics()
    
    def get_default_metrics(self) -> List[str]:
        """Get default metrics for the current evaluator."""
        return self.evaluator.default_metrics()

# Example usage
if __name__ == "__main__":
    # Sample data
    data = {
        "question": ["What is the capital of France?", "What is 2+2?"],
        "answer": ["Paris is the capital of France.", "The answer is 4."],
        "context": ["France is a country in Europe.", "Basic arithmetic operations."],
        "expected_answer": ["Paris", "4"]
    }
    
    # Initialize evaluator
    evaluator = LLMEvaluator(evaluator_type="ragas")
    
    # Run evaluation
    results = evaluator.evaluate(data)
    
    # Print results
    for idx, evals in results.items():
        print(f"\nExample {idx}:")
        for eval_result in evals:
            print(f"{eval_result.metric_name}: {eval_result.score:.3f} ({'PASS' if eval_result.passed else 'FAIL'})")
