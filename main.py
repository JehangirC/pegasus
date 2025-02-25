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
            
        self.evaluator_type = evaluator_type
            
        if evaluator_type == "ragas":
            self.evaluator = RagasEvaluator(metrics=metrics, threshold=threshold, llm=llm)
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

    def display_results(self, results: Dict[str, List[EvaluationResult]], title: str = "Evaluation Results"):
        """Display evaluation results in a rich formatted table with average scores."""
        console.print(Panel(title, style="bold magenta"))

        # Aggregate scores for each metric
        metric_scores = {}
        for idx, evals in results.items():
            for result in evals:
                if result.metric_name not in metric_scores:
                    metric_scores[result.metric_name] = []
                metric_scores[result.metric_name].append(result.score)

        # Calculate average scores
        average_scores = {
            metric: sum(scores) / len(scores)
            for metric, scores in metric_scores.items()
        }

        # Create table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="dim")
        table.add_column("Average Score", justify="right")
        table.add_column("Status", justify="center")

        # Populate table with average scores
        for metric, avg_score in average_scores.items():
            # Determine pass/fail status based on the threshold
            # Assuming you have a way to access the threshold for each metric
            # For example, you might store thresholds in a dictionary
            if self.evaluator_type == "ragas":
                from config import get_metric_threshold
                threshold = get_metric_threshold(metric, "ragas")
            elif self.evaluator_type == "deepeval":
                from config import get_metric_threshold
                threshold = get_metric_threshold(metric, "deepeval")
            else:
                threshold = 0.5  # Default threshold if evaluator type is unknown

            status = "[green]PASS" if avg_score >= threshold else "[red]FAIL"
            table.add_row(
                metric,
                f"{avg_score:.3f}",
                status,
            )

        console.print(table)

# Example usage
if __name__ == "__main__":
    from evaluator.deepeval_evaluator import DeepEvalEvaluator
    from evaluator.ragas_evaluator import RagasEvaluator
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore", message="Retrying langchain_google_vertexai.*", category=UserWarning)

    # Sample data
    # data = {
    # "question": ["What is the capital of France?", "What is the highest mountain?"],
    # "answer": ["Paris", "Mount Everest"],
    # "context": ["France is a country in Europe. Paris is its capital.", "Mount Everest is in the Himalayas."],
    # "expected_answer": ["Paris is the capital of France.", "Mount Everest is the highest mountain in the world."]
    # }
    # df = pd.DataFrame(data)
    
    from datasets import load_dataset

    amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True)
    df = pd.DataFrame(amnesty_qa["eval"].select(range(5)))

    # Initialize evaluator for Ragas
    ragas_evaluator = LLMEvaluator(evaluator_type="ragas", metrics=["answer_relevancy", "context_recall"] )

    # Run Ragas evaluation
    ragas_results = ragas_evaluator.evaluate(df)
    
    # Display Ragas results using rich formatting
    ragas_evaluator.display_results(ragas_results, "Ragas Evaluation Results")

    # Initialize evaluator for DeepEval
    deepeval_evaluator = LLMEvaluator(evaluator_type="deepeval", metrics=["answer_relevancy", "faithfulness"])

    try:
        deepeval_results = deepeval_evaluator.evaluate(df)
        # Display DeepEval results using rich formatting
        deepeval_evaluator.display_results(deepeval_results, "DeepEval Evaluation Results")
    except Exception as e:
        import traceback
        print(f"Error running DeepEval evaluation: {e}")
        print("Full traceback:")
        traceback.print_exc()
