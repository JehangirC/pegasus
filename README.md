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
- Google Cloud credentials configured
- Access to Vertex AI models (Gemini and Text_Embedding)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd pegasus/evaluator/pegasus
```

1. Install dependencies:

```bash
!chmod u+x ./pegasus_setup.sh
```

```bash
!./pegasus_setup.sh
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
                "name": "gemini-1.5-flash-002",  // Model for evaluation
                "config": {
                    "temperature": 0.0,
                    "top_k": 1
                }
            },
            "embeddings": {
                "name": "text-embedding-004"  // Embedding model
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
There is a `Getting_Started.ipynb` in `pegasus/evaluator/pegasus`.
```python
from evaluator.deepeval_evaluator import DeepEvalEvaluator
from evaluator.ragas_evaluator import RagasEvaluator
from main import LLMEvaluator
import pandas as pd

# Create your evaluation data
data = {
    "question": ["What is the capital of France?", "What is 2+2?"],
    "answer": ["Paris is the capital of France.", "The answer is 4."],
    "context": ["France is a country in Europe.", "Basic arithmetic operations."],
    "ground_truth": ["Paris", "4"]
}
df = pd.DataFrame(data)

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

ragas_results = ragas_evaluator.evaluate(df) #It needs to be a Pandas DataFrame
#ragas_evaluator.display_results(ragas_results, "Ragas Evaluation Results")
ragas_df = ragas_evaluator.to_df(ragas_results)
ragas_df
```

The required columns that need to be mapped are:

- `question`: The input question.
- `answer`: The model's response.
- `context`: The context provided to the model.

Optional columns:

- `ground_truth`: The expected answer (used by some metrics)

## Available Metrics

### RAGAS Metrics

- `answer_relevancy`: Evaluates how relevant the answer is to the question.
- `faithfulness`: Checks if the answer is supported by the provided context.
- `context_recall`: Measures how well the answer captures important information.
- `context_precision`: Evaluates the precision of information used from context.
- `answer_correctness`: Assesses factual correctness of the answer.
- `answer_similarity`: Measures similarity to expected answer.

### DeepEval Metrics

- `answer_relevancy`: Ealuates whether the prompt template in your generator is able to instruct your LLM to output relevant and helpful outputs based on the retrieval_context
- `faithfulness`: Evaluates whether the LLM used in your generator can output information that does not hallucinate AND contradict any factual information presented in the retrieval_context.
- `contextual_precision`: Measures precision of context usage
- `contextual_recall`: Evaluates recall of important context
- `bias`: Detects potential biases in responses
- `toxicity`: Checks for toxic or harmful content
For more information please checkout [https://docs.confident-ai.com/guides/guides-rag-evaluation](https://docs.confident-ai.com/guides/guides-rag-evaluation)

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
            reason="Why the score was given" # only for deepeval
        ),
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
