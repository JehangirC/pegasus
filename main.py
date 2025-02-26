"""Main interface for LLM evaluation."""

import logging
from typing import Dict, List, Optional, Union

import grpc
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from evaluator.base_evaluator import BaseEvaluator, EvaluationResult
from evaluator.config import get_metric_threshold
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.ragas_evaluator import RagasEvaluator

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

console = Console()


class LLMEvaluator:
    """Main class for evaluating LLM outputs."""

    def __init__(
        self,
        evaluator_type: str = "ragas",
        metrics: Optional[List[str]] = None,
        threshold: float = 0.0,
    ) -> None:
        """Initialize an LLM evaluator.

        Args:
            evaluator_type: Type of evaluator to use ('ragas' or 'deepeval')
            metrics: List of metrics to evaluate
            threshold: Overall threshold for pass/fail
        """
        if evaluator_type == "ragas":
            self.evaluator: BaseEvaluator = RagasEvaluator(
                metrics=metrics, threshold=threshold
            )
        elif evaluator_type == "deepeval":
            self.evaluator = DeepEvalEvaluator(metrics=metrics, threshold=threshold)
        else:
            raise ValueError(f"Unsupported evaluator type: {evaluator_type}")

    def evaluate(
        self, data: Union[pd.DataFrame, Dict[str, List[str]]]
    ) -> Dict[str, List[EvaluationResult]]:
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

    def display_results(
        self,
        results: Dict[str, List[EvaluationResult]],
        title: str = "Evaluation Results",
    ) -> None:
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

        # Define metrics where lower scores are better
        inverse_metrics = ["bias", "toxicity"]

        # Calculate average scores for each metric
        metric_scores: Dict[str, List[float]] = {}
        for example_results in results.values():
            for result in example_results:
                if result.metric_name not in metric_scores:
                    metric_scores[result.metric_name] = []
                metric_scores[result.metric_name].append(result.score)

        # Add rows to table
        for metric, scores in metric_scores.items():
            avg_score = sum(scores) / len(scores)
            evaluator_type = (
                "ragas" if isinstance(self.evaluator, RagasEvaluator) else "deepeval"
            )
            metric_threshold = get_metric_threshold(metric, evaluator_type)

            # For inverse metrics (bias/toxicity), lower is better
            if metric in inverse_metrics:
                passed = avg_score <= metric_threshold
                status = "PASS" if passed else "FAIL"
                style = "green" if passed else "red"
            else:
                # For regular metrics, higher is better
                passed = avg_score >= metric_threshold
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
        amnesty_qa = load_dataset(
            "explodinggradients/amnesty_qa", "english_v2", trust_remote_code=True
        )
        df = pd.DataFrame(amnesty_qa["eval"].select(range(1)))

        df.rename(columns={"contexts": "context"}, inplace=True)

        ragas_evaluator = LLMEvaluator(
            evaluator_type="ragas",
            metrics=[
                "answer_relevancy",
                "context_recall",
                "context_precision",
                "faithfulness",
                "answer_correctness",
            ],
        )

        ragas_results = ragas_evaluator.evaluate(df)
        ragas_evaluator.display_results(ragas_results, "Ragas Evaluation Results")

        deepeval_evaluator = LLMEvaluator(
            evaluator_type="deepeval",
            metrics=[
                "answer_relevancy",
                "faithfulness",
                "contextual_precision",
                "contextual_recall",
                "bias",
                "toxicity",
            ],
        )

        deepeval_results = deepeval_evaluator.evaluate(df)
        deepeval_evaluator.display_results(
            deepeval_results, "DeepEval Evaluation Results"
        )

    finally:
        # Ensure proper cleanup of gRPC resources
        try:
            grpc.aio.shutdown_asyncio_engine()
        except Exception as e:
            logging.debug(f"Error shutting down gRPC engine: {e}")
            pass
