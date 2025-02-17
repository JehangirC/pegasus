from typing import List, Dict, Any
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity,
)  # Import specific metrics
from datasets import Dataset
import pandas as pd
from .base_evaluator import BaseEvaluator, EvaluationInput, EvaluationResult

_SUPPORTED_RAGAS_METRICS = {  # Define supported metrics in a structured way
  "answer_relevancy": answer_relevancy,
  "faithfulness": faithfulness,
  "context_recall": context_recall,
  "context_precision": context_precision,
"context_relevancy": context_relevancy,
"answer_correctness": answer_correctness,
"answer_similarity": answer_similarity,
}
class RagasEvaluator(BaseEvaluator):
        """Evaluator using the Ragas library."""

        def __init__(self, metrics: List[str] = None):
            """
            Initializes the RagasEvaluator.

            Args:
                metrics: A list of metric names to use.  If None, uses a default set.
                         Must be a subset of the keys in _SUPPORTED_RAGAS_METRICS.
            """
            super().__init__()
            if metrics is None:
                metrics = ["answer_relevancy", "faithfulness"]  # A reasonable default set

            self.metrics = self._validate_metrics(metrics)


        def _validate_metrics(self, metrics: List[str]):
            """Validates and returns the Ragas metric objects."""
            validated_metrics = []
            for metric_name in metrics:
                if metric_name not in _SUPPORTED_RAGAS_METRICS:
                    raise ValueError(
                        f"Invalid Ragas metric: {metric_name}.  "
                        f"Supported metrics: {list(_SUPPORTED_RAGAS_METRICS.keys())}"
                    )
                validated_metrics.append(_SUPPORTED_RAGAS_METRICS[metric_name])
            return validated_metrics

        def evaluate(self, inputs: List[EvaluationInput]) -> List[Dict[str, Any]]:
            """Evaluates a list of inputs using Ragas."""

            # Convert EvaluationInputs to a format Ragas expects (Dataset).
            # This is crucial for correct data handling.

            data = {
              "question": [],
              "answer": [],
              "contexts": [],
              "ground_truths": []
            }

            for inp in inputs:
                data["question"].append(inp.question)
                data["answer"].append(inp.answer)
                data["contexts"].append(inp.context if isinstance(inp.context, list) else [inp.context]) #Ragas needs a list of strings
                data["ground_truths"].append(inp.expected_answer if inp.expected_answer else [])

            # Convert to a Dataset
            dataset = Dataset.from_pandas(pd.DataFrame(data))

            # Run Ragas evaluation
            result = evaluate(dataset, metrics=self.metrics)

            # Convert the Ragas result (which is a Dataset) to a list of dictionaries.
            results_list = []
            df = result.to_pandas()
            for i in range(len(inputs)):
                res_dict = {}
                for metric in self.metrics:
                  res_dict[metric.name] = df.iloc[i][metric.name]
                results_list.append(res_dict)

            return results_list

        def supported_metrics(self) -> List[str]:
          return list(_SUPPORTED_RAGAS_METRICS.keys())