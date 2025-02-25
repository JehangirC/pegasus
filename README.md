# Pegasus

A powerful evaluation framework for Large Language Models using RAGAS and DeepEval metrics, with native support for Google Cloud's Vertex AI.

## Features

- Dual evaluation frameworks:
  - RAGAS metrics for comprehensive RAG evaluation
  - DeepEval metrics for general LLM evaluation
- Native Google Cloud Vertex AI integration
- Configurable via JSON
- Support for all major evaluation metrics
- Pandas DataFrame interface for easy data handling

## Prerequisites

- Python 3.8+
- Google Cloud account with Vertex AI API enabled
- Google Cloud credentials configured
- Access to Vertex AI models (Gemini and Gecko)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pegasus
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Set up Google Cloud credentials:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

## Configuration

The evaluator is configured via `config.json`. Here's an explanation of the configuration options:

```json
{
    "vertex_ai": {
        "project_id": null,  // Will use GOOGLE_CLOUD_PROJECT env var if null
        "location": "europe-west2",  // Google Cloud region
        "models": {
            "llm": {
                "name": "gemini-1.5-flash",  // Model for evaluation
                "config": {
                    "temperature": 0.0,
                    "top_k": 1
                }
            },
            "embeddings": {
                "name": "textembedding-gecko@003"  // Embedding model
            }
        }
    },
    "metrics": {
        "ragas": {
            "default": [
                "answer_relevancy",     // Measures how relevant the answer is
                "faithfulness",         // Checks if answer is supported by context
                "context_recall",       // Measures information capture
                "context_precision",    // Measures precision of used context
                "answer_correctness",   // Evaluates factual accuracy
                "answer_similarity"     // Compares to expected answer
            ]
        },
        "deepeval": {
            "default": [
                "answer_relevancy",     // Similar to RAGAS
                "faithfulness",         // Similar to RAGAS
                "contextual_precision", // Context usage precision
                "contextual_recall",    // Important context coverage
                "bias",                // Checks for biased responses
                "toxicity"             // Checks for harmful content
            ]
        }
    }
}
```

## Usage

### Basic Usage

```python
from pegasus import LLMEvaluator
from pegasus.llms.vertexai_llm import VertexAILLM
import pandas as pd

# Create your evaluation data
data = {
    "question": ["What is the capital of France?", "What is 2+2?"],
    "answer": ["Paris is the capital of France.", "The answer is 4."],
    "contexts": ["France is a country in Europe.", "Basic arithmetic operations."],
    "expected_answer": ["Paris", "4"]
}
df = pd.DataFrame(data)

# Create a custom LLM (optional for Ragas, required for DeepEval)
custom_llm = VertexAILLM(
    model_name="gemini-1.5-flash",
    project_id="your-project",
    location="europe-west2"
)

# Initialize evaluator with Ragas (will use config defaults if no LLM provided)
evaluator = LLMEvaluator(evaluator_type="ragas")
# Or provide a custom LLM
evaluator = LLMEvaluator(evaluator_type="ragas", llm=custom_llm)

# Initialize evaluator with DeepEval (requires LLM)
evaluator = LLMEvaluator(evaluator_type="deepeval", llm=custom_llm)

# Run evaluation
results = evaluator.evaluate(df)
```

### Using Custom Metrics

You can override the default metrics from the config by passing either a single metric or a list of metrics:

```python
# Using a single metric
evaluator = LLMEvaluator(
    evaluator_type="ragas",
    metrics="answer_relevancy",
    llm=custom_llm  # optional for Ragas
)

# Using multiple metrics
evaluator = LLMEvaluator(
    evaluator_type="deepeval",
    metrics=["answer_relevancy", "faithfulness"],
    llm=custom_llm  # required for DeepEval
)

# With other parameters
evaluator = LLMEvaluator(
    evaluator_type="ragas",
    metrics=["answer_relevancy", "context_precision"],
    threshold=0.7,
    llm=custom_llm  # optional
)
```

The specified metrics will completely replace the default metrics defined in your configuration. Make sure to only use metrics that are supported by your chosen evaluator type (see Available Metrics section below).

### Using Custom LLM

```python
from pegasus.llms.vertexai_llm import VertexAILLM

custom_llm = VertexAILLM(
    model_name="gemini-1.5-flash",
    project_id="your-project",
    location="europe-west2"
)

evaluator = LLMEvaluator(
    evaluator_type="deepeval",
    llm=custom_llm
)
```

### Using Custom Column Names

You can map your DataFrame columns to the required column names using the `column_mapping` parameter:

```python
# Define your column mapping
column_mapping = {
    "question": "user_query",      # Maps 'user_query' to required 'question' column
    "answer": "response",          # Maps 'response' to required 'answer' column
    "context": "sources",          # Maps 'sources' to required 'context' column
    "ground_truth": "expected"     # Maps 'expected' to optional 'ground_truth' column
}

# Create your evaluation data with custom column names
data = {
    "user_query": ["What is the capital of France?"],
    "response": ["Paris is the capital of France."],
    "sources": ["France is a country in Europe."],
    "expected": ["Paris"]
}
df = pd.DataFrame(data)

# Initialize evaluator with column mapping
evaluator = LLMEvaluator(
    evaluator_type="ragas",
    column_mapping=column_mapping
)

# Run evaluation
results = evaluator.evaluate(df)
```

The required columns that need to be mapped are:

- `question`: The input question
- `answer`: The model's response
- `context`: The context provided to the model

Optional columns:

- `ground_truth`: The expected answer (used by some metrics)

## Available Metrics

### RAGAS Metrics

- `answer_relevancy`: Evaluates how relevant the answer is to the question
- `faithfulness`: Checks if the answer is supported by the provided context
- `context_recall`: Measures how well the answer captures important information
- `context_precision`: Evaluates the precision of information used from context
- `answer_correctness`: Assesses factual correctness of the answer
- `answer_similarity`: Measures similarity to expected answer

### DeepEval Metrics

- `answer_relevancy`: Similar to RAGAS metric
- `faithfulness`: Similar to RAGAS metric
- `contextual_precision`: Measures precision of context usage
- `contextual_recall`: Evaluates recall of important context
- `bias`: Detects potential biases in responses
- `toxicity`: Checks for toxic or harmful content

## Output Format

The evaluator returns a dictionary with evaluation results for each input row:

```python
{
    0: [  # Index of the input row
        EvaluationResult(
            metric_name="answer_relevancy",
            score=0.95,
            passed=True,
            threshold=0.5,
            explanation="Score explanation",
            reason="Why the score was given"
        ),
        # More metric results...
    ],
    # More rows...
}
```

## Development

### Running Tests

```bash
pytest tests/
```

### Contributing

1. Fork the repository
2. Create your feature branch
3. Write tests for your changes
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- RAGAS evaluation framework
- DeepEval framework
