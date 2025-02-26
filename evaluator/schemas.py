"""Configuration schemas and loading utilities."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable

from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    """Configuration for a Vertex AI model."""

    name: str
    config: Optional[Dict[str, Union[float, int]]] = Field(default_factory=lambda: {})


class VertexAIConfig(BaseModel):
    """Configuration for Vertex AI."""

    project_id: Optional[str] = None
    location: str = "europe-west2"
    models: Dict[str, ModelConfig]


class ThresholdConfig(BaseModel):
    """Configuration for metric thresholds."""

    default: float = 0.5
    answer_relevancy: Optional[float] = None
    faithfulness: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_similarity: Optional[float] = None
    contextual_precision: Optional[float] = None
    contextual_recall: Optional[float] = None
    bias: Optional[float] = None
    toxicity: Optional[float] = None

    def get_threshold(self, metric_name: str) -> float:
        """Get threshold for a specific metric."""
        return getattr(self, metric_name, self.default)


class MetricConfig(BaseModel):
    """Configuration for a metric type."""

    default: List[str]
    thresholds: ThresholdConfig


class LogLevel(str, Enum):
    """Valid logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseModel):
    """Configuration for logging."""

    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling."""

    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: int = 30

    @validator("max_retries")
    def validate_max_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_retries must be non-negative")
        return v

    @validator("retry_delay")
    def validate_retry_delay(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("retry_delay must be positive")
        return v

    @validator("timeout")
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("timeout must be positive")
        return v


class Config(BaseModel):
    """Root configuration."""

    vertex_ai: VertexAIConfig
    metrics: Dict[str, MetricConfig]
    logging: LoggingConfig
    error_handling: ErrorHandlingConfig
