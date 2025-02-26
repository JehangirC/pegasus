"""Utility functions for the evaluator."""

import json
import logging
from functools import wraps
from time import sleep, time
from typing import Any, Callable, Dict, List, Optional, TypeVar

import pandas as pd

from .base_evaluator import EvaluationInput
from .config import CONFIG
from .constants import DataColumns, ErrorMessages

T = TypeVar('T')

def setup_logger(name: str) -> logging.Logger:
    """Set up a logger with the configured settings.

    Args:
        name: Name of the logger

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(CONFIG.logging.level.value)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create handler with configured format
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(CONFIG.logging.format))
    logger.addHandler(handler)

    return logger


def validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that DataFrame has required columns.

    Args:
        df: DataFrame to validate

    Raises:
        ValueError: If required columns are missing
    """
    required_columns = [DataColumns.QUESTION, DataColumns.ANSWER, DataColumns.CONTEXT]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(ErrorMessages.INVALID_DATA.format(columns=missing))


def retry_on_exception(max_retries: Optional[int] = None, delay: Optional[float] = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to retry function on exception.

    Args:
        max_retries: Maximum number of retries (default from config)
        delay: Delay between retries in seconds (default from config)

    Returns:
        A decorator function that adds retry functionality
    """
    max_retries_val = max_retries if max_retries is not None else CONFIG.error_handling.max_retries
    delay_val = delay if delay is not None else CONFIG.error_handling.retry_delay

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries_val + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries_val:
                        sleep(delay_val)
                    continue
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Unexpected state in retry_on_exception")
        return wrapper
    return decorator


def time_execution(logger: logging.Logger) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to log function execution time.

    Args:
        logger: Logger instance to use for logging

    Returns:
        A decorator function that adds execution time logging
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.debug(f"{func.__name__} executed in {end - start:.2f} seconds")
            return result
        return wrapper
    return decorator


def load_eval_data_json(filepath: str) -> List[EvaluationInput]:
    """Loads evaluation data from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of EvaluationInput objects
    """
    with open(filepath, "r") as f:
        data = json.load(f)
        validated_data = [EvaluationInput(**item) for item in data]
    return validated_data


def load_eval_data_csv(filepath: str) -> List[EvaluationInput]:
    """Loads evaluation data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of EvaluationInput objects
        
    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(filepath)

    required_columns = ["question", "answer", "context"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            f"CSV file must contain columns: {', '.join(required_columns)}"
        )
    if "ground_truths" not in df.columns:
        df["ground_truths"] = ""
    records: List[Dict[str, Any]] = df.to_dict(orient="records")  # type: ignore[assignment]
    return [EvaluationInput(**row) for row in records]

# Create main logger for the package
logger = setup_logger("llm_evaluator")
