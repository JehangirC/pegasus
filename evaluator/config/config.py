"""Configuration module for LLM evaluator."""
import json
import logging
from pathlib import Path
from functools import lru_cache
from ..schemas import Config

class ConfigurationError(Exception):
    """Raised when there's an error in configuration."""
    pass

@lru_cache()
def load_config() -> Config:
    """Load configuration from config.json file.
    
    Returns:
        Config: Validated configuration object
        
    Raises:
        ConfigurationError: If config file cannot be loaded or is invalid
    """
    try:
        config_path = Path(__file__).parent.parent.parent / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        return Config(**config_dict)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Failed to load configuration file: {str(e)}")
    except ValueError as e:
        raise ConfigurationError(f"Invalid configuration: {str(e)}")

# Load configuration
CONFIG = load_config()

# Vertex AI Configuration
PROJECT_ID = CONFIG.vertex_ai.project_id
LOCATION = CONFIG.vertex_ai.location
VERTEX_MODELS = {k: v.dict() for k, v in CONFIG.vertex_ai.models.items()}

# Metrics Configuration
DEFAULT_RAGAS_METRICS = CONFIG.metrics["ragas"].default
DEFAULT_DEEPEVAL_METRICS = CONFIG.metrics["deepeval"].default

def get_metric_threshold(metric_name: str, evaluator_type: str) -> float:
    """Get the threshold for a specific metric.
    
    Args:
        metric_name: Name of the metric
        evaluator_type: Type of evaluator ('ragas' or 'deepeval')
        
    Returns:
        float: Threshold value for the metric
    """
    return CONFIG.metrics[evaluator_type].thresholds.get_threshold(metric_name)

# Configure logging
logging.basicConfig(
    level=getattr(logging, CONFIG.logging.level.value),
    format=CONFIG.logging.format)