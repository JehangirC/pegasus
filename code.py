from evaluator.ragas_evaluator import RagasEvaluator
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.utils import load_eval_data_json, load_eval_data_csv
from evaluator.base_evaluator import EvaluationInput

# Example using dummy data
inputs = [
    EvaluationInput(
        question="What is the capital of France?",
        answer="The capital of France is Paris.",
        context="France is a country in Europe. Its capital is Paris.",
    ),
    EvaluationInput(
        question="Who painted the Mona Lisa?",
        answer="The Mona Lisa was painted by Leonardo da Vinci.",
        context="Leonardo da Vinci was an Italian Renaissance artist.",
    ),
    EvaluationInput(
        question="What is the highest mountain in the world?",
        answer="Mount Everest is not the highest mountain.",  # Incorrect answer
        context="Mount Everest is located in the Himalayas.",
    ),
]

# --- Ragas Evaluation ---
ragas_evaluator = RagasEvaluator(
    metrics=["answer_relevancy", "faithfulness", "context_recall"]
)
ragas_results = ragas_evaluator.evaluate(inputs)
print("Ragas Results:", ragas_results)


# --- DeepEval Evaluation ---
deepeval_evaluator = DeepEvalEvaluator(
    metrics=["answer_relevancy", "faithfulness"]
)
deepeval_results = deepeval_evaluator.evaluate(inputs)
print("DeepEval Results:", deepeval_results)

# --- Example using data loading (assuming you have data.json) ---

# Load data using the utility function
# try:
#     eval_data = load_eval_data_json("data.json")
#     ragas_results_loaded = ragas_evaluator.evaluate(eval_data)
#     print("Ragas Results (Loaded Data):", ragas_results_loaded)

#     deepeval_results_loaded = deepeval_evaluator.evaluate(eval_data)
#     print("DeepEval Results (loaded Data):", deepeval_results_loaded)
# except FileNotFoundError:
#     print("data.json not found.  Skipping data loading example.")
# except ValueError as e:
#     print(f"Error loading data: {e}")


#--- Example using csv loading (assuming you have data.csv)
# try:
#   eval_data = load_eval_data_csv("data.csv")
#   ragas_results_loaded = ragas_evaluator.evaluate(eval_data)
#   print("Ragas Results (Loaded Data):", ragas_results_loaded)

#   deepeval_results_loaded = deepeval_evaluator.evaluate(eval_data)
#   print("DeepEval Results (loaded Data):", deepeval_results_loaded)

# except FileNotFoundError:
#   print("data.csv not found. Skipping data loading example")
# except ValueError as e:
#   print(f"Error loading data: {e}")


#--- Example using EvaluationResult
# result = EvaluationResult(metric_name="my_metric", score=0.8)
# print(result)
# print(result.json()) #convert to json