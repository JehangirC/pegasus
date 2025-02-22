"""Utility functions for the evaluator."""
import json
import logging
from typing import List, Dict, Any
from functools import wraps
from time import time
from .config import CONFIG
from .constants import DataColumns, ErrorMessages
from .base_evaluator import EvaluationInput
import pandas as pd

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

def retry_on_exception(max_retries: int = None, delay: float = None):
    """Decorator to retry function on exception.
    
    Args:
        max_retries: Maximum number of retries (default from config)
        delay: Delay between retries in seconds (default from config)
    """
    max_retries = max_retries or CONFIG.error_handling.max_retries
    delay = delay or CONFIG.error_handling.retry_delay
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        time.sleep(delay)
                    continue
            raise last_exception
        return wrapper
    return decorator

def time_execution(logger: logging.Logger):
    """Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            result = func(*args, **kwargs)
            end = time()
            logger.debug(f"{func.__name__} executed in {end - start:.2f} seconds")
            return result
        return wrapper
    return decorator

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

# Create main logger for the package
logger = setup_logger("llm_evaluator")