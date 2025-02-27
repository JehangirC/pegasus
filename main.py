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
    TimeRemainingColumn,
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

        # Create a progress display that works in both terminal and notebooks
        console.print()  # Add blank line before progress

        # Determine number of steps based on evaluator type and metrics count
        num_metrics = len(self.evaluator.metrics)
        num_examples = len(df)
        total_steps = num_metrics * num_examples

        # Create a progress bar with better timing information
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            SpinnerColumn(),
            console=console,
            refresh_per_second=5,  # Lower refresh rate for better notebook compatibility
            transient=False,  # Ensures the progress bar stays visible
            expand=True,  # Use full width
        ) as progress:
            main_task = progress.add_task(
                f"[cyan]Running {self.evaluator.__class__.__name__} evaluation...",
                total=total_steps,
            )

            # Create a wrapper for the evaluator that updates progress
            # For DeepEval, we'll manually update per metric since it processes each metric separately
            if isinstance(self.evaluator, DeepEvalEvaluator):
                # Create a dictionary to store results
                evaluation_results: Dict[str, List[EvaluationResult]] = {}

                # Process each example and update progress manually
                for idx, row in df.iterrows():
                    # Format a sub-task description that shows which example is being processed
                    progress.update(
                        main_task,
                        description=f"[cyan]Evaluating example {int(idx)+1}/{num_examples}",
                    )

                    # Let the actual evaluator process this example
                    results = self.evaluator.evaluate(pd.DataFrame([row]))

                    # Store results
                    evaluation_results[str(idx)] = results.get("0", [])

                    # Update progress based on metrics count
                    progress.update(main_task, advance=num_metrics)

                results = evaluation_results
            else:
                # For Ragas, we update once at the end as it processes all at once
                progress.update(
                    main_task, description="[cyan]Running Ragas evaluation..."
                )
                results = self.evaluator.evaluate(df)
                progress.update(main_task, completed=total_steps)

            # Complete the progress bar but without a final message that might clash with DeepEval output
            progress.update(main_task, completed=total_steps)

            console.print()  # Add blank line after progress
            return results

    def get_supported_metrics(self) -> List[str]:
        """Get list of metrics supported by the current evaluator."""
        return self.evaluator.supported_metrics()

    def get_default_metrics(self) -> List[str]:
        """Get default metrics for the current evaluator."""
        return self.evaluator.default_metrics()

    def to_df(self, results: Dict[str, List[EvaluationResult]]) -> pd.DataFrame:
        """
        Aggregate evaluation results with metrics as columns.

        Parameters:
        -----------
        results : dict
            Dictionary of evaluation results where keys are IDs and values are lists of result objects.

        Returns:
        --------
        pd.DataFrame
            DataFrame with metrics as columns and a single row showing average scores across all IDs.
        """
        # Create a dictionary to store aggregated scores by metric
        metric_scores = {}

        for _key, eval_results in results.items():
            for result in eval_results:
                metric_name = result.metric_name
                score = result.score

                if metric_name not in metric_scores:
                    metric_scores[metric_name] = {
                        "total_score": 0.0,
                        "count": 0,
                        "passed": 0,
                        "threshold": result.threshold,
                        "explanation": result.explanation,
                    }

                # Convert score to float explicitly to avoid type errors
                metric_scores[metric_name]["total_score"] += float(score)  # type: ignore[operator]
                metric_scores[metric_name]["count"] += 1
                if result.passed:
                    metric_scores[metric_name]["passed"] += 1

        # Calculate averages and create the DataFrame
        avg_data = {"Metric": "Average Across All IDs"}

        for metric, stats in metric_scores.items():
            # Ensure we're working with float values for division
            total_score = float(stats["total_score"])  # type: ignore[arg-type]
            count = float(stats["count"])  # type: ignore[arg-type]
            avg_score = total_score / count if count > 0 else 0.0
            avg_data[metric] = round(avg_score, 3)  # type: ignore[assignment]
            # You could also include pass rate if needed
            # avg_data[f"{metric}_pass_rate"] = stats['passed'] / stats['count']

        df_avg = pd.DataFrame([avg_data])

        # Set 'Metric' as the index for better display
        df_avg = df_avg.set_index("Metric")

        return df_avg

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
            # Ensure we're working with float values for calculations
            float_scores = [float(score) for score in scores]
            avg_score = sum(float_scores) / len(float_scores) if float_scores else 0.0
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
