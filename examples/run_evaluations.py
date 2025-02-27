"""Example usage of the LLM Evaluator."""
import logging

import pandas as pd

from ..main import LLMEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

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

        print("Running RAGAS evaluation...")
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

        print("Running DeepEval evaluation...")
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
            import grpc

            grpc.aio.shutdown_asyncio_engine()
        except Exception as e:
            logging.debug(f"Error shutting down gRPC engine: {e}")
            pass
