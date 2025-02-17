import json
from typing import List, Dict, Any
from .base_evaluator import EvaluationInput
import pandas as pd

def load_eval_data_json(filepath: str) -> List[EvaluationInput]:
    """Loads evaluation data from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
        # Validate that the data has the expected structure.
        validated_data = [EvaluationInput(**item) for item in data] # Uses the pydantic model
    return validated_data

def load_eval_data_csv(filepath:str) -> List[EvaluationInput]:
    """Loads evaluation data from a CSV file."""
    df = pd.read_csv(filepath)

    # Check if required columns exist
    required_columns = ['question', 'answer', 'context']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV file must contain columns: {', '.join(required_columns)}")
    if 'ground_truths' not in df.columns:
        df['ground_truths'] = ''
    # Convert DataFrame rows to EvaluationInput objects
    return [EvaluationInput(**row) for row in df.to_dict('records')]


# def save_results_json(results: List[Dict[str, Any]], filepath: str):
#     """Saves evaluation results to a JSON file."""
#     with open(filepath, "w") as f:
#         json.dump(results, f, indent=4)

# Add other utility functions as needed (e.g., for loading configurations).