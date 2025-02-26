"""Constants used throughout the evaluator."""

from enum import Enum


class EvaluatorType(str, Enum):
    """Types of evaluators available."""

    RAGAS = "ragas"
    DEEPEVAL = "deepeval"


# Column names for evaluation data
class DataColumns:
    """Column names for input data."""

    QUESTION = "question"
    ANSWER = "answer"
    CONTEXT = "context"
    EXPECTED_ANSWER = "expected_answer"


# Default values
ASYNC_MODE = False
RESPONSE_VALIDATION = False


# Error messages
class ErrorMessages:
    """Error messages used in the evaluator."""

    INVALID_METRIC = "Invalid metric: {metric}. Supported metrics: {supported}"
    MISSING_LLM = "LLM must be provided for evaluation"
    INVALID_DATA = "Input data must contain required columns: {columns}"
    CONFIG_LOAD_ERROR = "Failed to load configuration: {error}"
    EVALUATION_ERROR = "Error during evaluation: {error}"


# Success messages
class SuccessMessages:
    """Success messages used in the evaluator."""

    EVALUATION_COMPLETE = "Evaluation completed successfully"
    CONFIG_LOADED = "Configuration loaded successfully"
    LLM_INITIALIZED = "LLM initialized successfully"
