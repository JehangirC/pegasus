"""Configuration module for LLM evaluator."""
import os
import json
import logging
from typing import Dict, Any
from pathlib import Path
from functools import lru_cache

class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

@lru_cache()
def load_config() -> Dict[str, Any]:
    """Load configuration from config.json file.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
        
    Raises:
        ConfigurationError: If config file cannot be loaded or is invalid
    """
    try:
        config_path = Path(__file__).parent / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Override with environment variables if present
        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
            config["vertex_ai"]["project_id"] = os.environ["GOOGLE_CLOUD_PROJECT"]
        if os.environ.get("GOOGLE_CLOUD_LOCATION"):
            config["vertex_ai"]["location"] = os.environ["GOOGLE_CLOUD_LOCATION"]
        
        return config
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")

# Load configuration
CONFIG = load_config()

# Vertex AI Configuration
PROJECT_ID = CONFIG["vertex_ai"]["project_id"]
LOCATION = CONFIG["vertex_ai"]["location"]
VERTEX_MODELS = CONFIG["vertex_ai"]["models"]

# Metrics Configuration
RAGAS_CONFIG = CONFIG["metrics"]["ragas"]
DEEPEVAL_CONFIG = CONFIG["metrics"]["deepeval"]
DEFAULT_RAGAS_METRICS = RAGAS_CONFIG["default"]
DEFAULT_DEEPEVAL_METRICS = DEEPEVAL_CONFIG["default"]

def get_metric_threshold(metric_name: str, evaluator_type: str) -> float:
    """Get the threshold for a specific metric.
    
    Args:
        metric_name: Name of the metric
        evaluator_type: Type of evaluator ('ragas' or 'deepeval')
        
    Returns:
        float: Threshold value for the metric
    """
    config = CONFIG["metrics"][evaluator_type]["thresholds"]
    return config.get(metric_name, config["default"])

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG["logging"]["level"]),
    format=CONFIG["logging"]["format"]
)