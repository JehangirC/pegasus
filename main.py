"""Main interface for LLM evaluation."""
import os
import google.auth
import google.auth.transport.requests
import pandas as pd
from typing import List, Dict, Any, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
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

console = Console()

class LLMEvaluator:
    """Main class for evaluating LLM outputs."""
    
    def __init__(self, evaluator_type: str = "ragas", metrics: Union[str, List[str]] = None, threshold: float = 0.5, llm=None):
        """Initialize the evaluator.
        
        Args:
            evaluator_type: Type of evaluator to use ("ragas" or "deepeval")
            metrics: Single metric or list of metrics to evaluate. If None, uses default metrics
            threshold: Score threshold for passing evaluation
            llm: Language model to use for evaluation (required for DeepEval, optional for Ragas)
        """
        # Convert single metric to list
        if isinstance(metrics, str):
            metrics = [metrics]
            
        if evaluator_type == "ragas":
            self.evaluator = RagasEvaluator(metrics=metrics, threshold=threshold, llm=llm)
        elif evaluator_type == "deepeval":
            if llm is None:
                raise ValueError("DeepEval requires an LLM instance for evaluation")
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

    def display_results(self, results: Dict[str, List[EvaluationResult]], title: str = "Evaluation Results"):
        """Display evaluation results in a rich formatted table."""
        console.print(Panel(title, style="bold magenta"))
        
        for idx, evals in results.items():
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Metric", style="dim")
            table.add_column("Score", justify="right")
            table.add_column("Status", justify="center")
            table.add_column("Explanation", style="dim")
            
            for result in evals:
                status = "[green]PASS" if result.passed else "[red]FAIL"
                table.add_row(
                    result.metric_name,
                    f"{result.score:.3f}",
                    status,
                    Text(result.explanation, style="italic")
                )
            
            console.print(f"\n[bold blue]Example {idx}:")
            console.print(table)

# Example usage
if __name__ == "__main__":
    from evaluator.llms.vertexai_llm import VertexAILLM
    from main import LLMEvaluator
    import pandas as pd

    # Sample data
    data = {
        "question": ["What is the capital of France?", "What is 2+2?"],
        "answer": ["Paris is the capital of France.", "The answer is 4."],
        "context": ["France is a country in Europe.", "Basic arithmetic operations."],
        "expected_answer": ["Paris", "4"]
    }
    df = pd.DataFrame(data)

    # Initialize evaluator for Ragas
    ragas_evaluator = LLMEvaluator(evaluator_type="ragas")

    # Run Ragas evaluation
    ragas_results = ragas_evaluator.evaluate(df)
    
    # Display Ragas results using rich formatting
    ragas_evaluator.display_results(ragas_results, "Ragas Evaluation Results")

    # Initialize the LLM (Language Model)
    llm = VertexAILLM(model_name="gemini-1.5-flash", project_id="testing-ragas", location="europe-west2")

    # Initialize evaluator for DeepEval
    deepeval_evaluator = LLMEvaluator(evaluator_type="deepeval", llm=llm)

    # Run DeepEval evaluation
    deepeval_results = deepeval_evaluator.evaluate(df)
    
    # Display DeepEval results using rich formatting
    deepeval_evaluator.display_results(deepeval_results, "DeepEval Evaluation Results")
