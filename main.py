"""Main interface for LLM evaluation."""
import pandas as pd
import grpc
import asyncio
from typing import List, Dict, Any, Union
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from evaluator.ragas_evaluator import RagasEvaluator
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.llms.vertexai_llm import VertexAILLM
from evaluator.base_evaluator import EvaluationInput, EvaluationResult
from evaluator.llms.base_llm import BaseLLM
import logging

logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

console = Console()

class LLMEvaluator:
    """Main class for evaluating LLM outputs."""

    def __init__(self, evaluator_type: str = "ragas", metrics: List[str] = None, threshold: float = 0.0):
        """Initialize an LLM evaluator.
        
        Args:
            evaluator_type: Type of evaluator to use ('ragas' or 'deepeval')
            metrics: List of metrics to evaluate
            threshold: Overall threshold for pass/fail
            column_mapping: Dictionary mapping required column names to input column names
        """
        if evaluator_type == "ragas":
            self.evaluator = RagasEvaluator(metrics=metrics, threshold=threshold)
        elif evaluator_type == "deepeval":
            self.evaluator = DeepEvalEvaluator(metrics=metrics, threshold=threshold)
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

    def evaluate(self, data: Union[pd.DataFrame, Dict[str, List[str]]]) -> Dict[str, List[EvaluationResult]]:
        """Evaluate LLM outputs.
        
        Args:
            data: DataFrame or dictionary with required columns/keys.
                 Column names can be mapped using column_mapping in __init__
                
        Returns:
            Dictionary mapping example indices to their evaluation results
        """
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        else:
            df = data.copy()

        console.print()
        console.print()  # Add extra blank lines before progress
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            SpinnerColumn(),
            console=console,
            transient=False,  # This ensures the progress bar stays visible
        ) as progress:
            task = progress.add_task("[cyan]Evaluating...", total=1)
            results = self.evaluator.evaluate(df)
            progress.update(task, advance=1, description="[green]Evaluation complete")
            console.print()  # Add blank line after progress
            return results

    def get_supported_metrics(self) -> List[str]:
        """Get list of metrics supported by the current evaluator."""
        return self.evaluator.supported_metrics()
    
    def get_default_metrics(self) -> List[str]:
        """Get default metrics for the current evaluator."""
        return self.evaluator.default_metrics()
    
    def display_results(self, results: Dict[str, List[EvaluationResult]], title: str = "Evaluation Results"):
        """Display evaluation results using rich formatting.
        
        Args:
            results: Dictionary of evaluation results from evaluate()
            title: Title to display above results table
        """
        # Create results table
        table = Table()
        table.add_column("Metric", justify="left")
        table.add_column("Average Score", justify="right")
        table.add_column("Status", justify="center")
        
        # Calculate average scores for each metric
        metric_scores = {}
        for example_results in results.values():
            for result in example_results:
                if result.metric_name not in metric_scores:
                    metric_scores[result.metric_name] = []
                metric_scores[result.metric_name].append(result.score)
        
        # Add rows to table
        for metric, scores in metric_scores.items():
            avg_score = sum(scores) / len(scores)
            passed = avg_score >= self.evaluator.threshold
            status = "PASS" if passed else "FAIL"
            style = "green" if passed else "red"
            table.add_row(metric, f"{avg_score:.3f}", Text(status, style=style))
            
        # Display results
        console.print(Panel(table, title=title, style="magenta bold"))

# Example usage
if __name__ == "__main__":

    try:
        from datasets import load_dataset
        # Example with default column names
        amnesty_qa = load_dataset("explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True)
        df = pd.DataFrame(amnesty_qa["eval"].select(range(5)))

        df.rename(columns={"contexts":"context"}, inplace=True) 

        ragas_evaluator = LLMEvaluator(
            evaluator_type="ragas", 
            metrics=["answer_relevancy", "context_recall"]
        )

        # Run Ragas evaluation on custom columns
        ragas_results = ragas_evaluator.evaluate(df)
        ragas_evaluator.display_results(ragas_results, "Ragas Evaluation Results (Custom Columns)")

        # Initialize evaluator for DeepEval (standard columns)
        deepeval_evaluator = LLMEvaluator(
            evaluator_type="deepeval", 
            metrics=["answer_relevancy", "faithfulness"]
        )

        try:
            deepeval_results = deepeval_evaluator.evaluate(df)
            deepeval_evaluator.display_results(deepeval_results, "DeepEval Evaluation Results")
        except Exception as e:
            import traceback
            print(f"Error running DeepEval evaluation: {e}")
            print("Full traceback:")
            traceback.print_exc()
    finally:
        # Ensure proper cleanup of gRPC resources
        try:
            grpc.aio.shutdown_asyncio_engine()
        except:
            pass
