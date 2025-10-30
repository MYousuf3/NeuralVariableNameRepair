# Neural Variable Name Repair - Pipeline

## Overview

This pipeline loads masked code data, prepares it for LLM inference, and evaluates predictions using exact string matching.

## Core Files

- **`data_pipeline.py`** - Loads and formats data for LLM input
- **`evaluate_predictions.py`** - Evaluates LLM predictions and calculates accuracy
- **`example_output.jsonl`** - Sample extracted function data
- **`extract_functions.py`** - Extracts and masks functions from code

## Quick Usage

### 1. Load Data

```python
from data_pipeline import DataPipeline

pipeline = DataPipeline('example_output.jsonl')
pipeline.load_data()

# Get data as separate arrays
input_texts, target_texts = pipeline.get_separate_arrays()

# Or get as paired format
paired_data = pipeline.get_paired_format()

# Or get LLM-ready prompts
prompts = pipeline.prepare_llm_prompts()
```

### 2. Get LLM Predictions

```python
# Replace with your actual LLM API call
llm_responses = []
for input_text in input_texts:
    response = your_llm_api(input_text)  # Your LLM call here
    llm_responses.append(response)
```

### 3. Evaluate

```python
from evaluate_predictions import PredictionEvaluator

evaluator = PredictionEvaluator()
cleaned_responses = evaluator.prepare_llm_responses(llm_responses)
metrics = evaluator.evaluate(cleaned_responses, target_texts)

# Display results
evaluator.print_report(metrics)

# Save to file
evaluator.save_results(metrics, 'results.json')
```

## Complete Example

```python
from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator

# Load data
pipeline = DataPipeline('example_output.jsonl')
pipeline.load_data()
input_texts, target_texts = pipeline.get_separate_arrays()

# Get LLM predictions
llm_responses = [your_llm(text) for text in input_texts]

# Evaluate
evaluator = PredictionEvaluator()
cleaned = evaluator.prepare_llm_responses(llm_responses)
metrics = evaluator.evaluate(cleaned, target_texts)

# Results
evaluator.print_report(metrics)
print(f"Accuracy: {metrics['accuracy_percentage']:.2f}%")
```

## Scoring

- **1/1** if prediction exactly matches target
- **0/1** if any difference
- Final accuracy = correct_count / total_trials

## Output

Console output shows:
- Total trials
- Correct/Incorrect counts
- Accuracy percentage
- Trial-by-trial breakdown

JSON file includes:
- Summary metrics
- Detailed per-trial results with predictions and targets

## Data Formats

Input JSONL format:
```json
{
  "file": "row_000001",
  "func_name": "id",
  "input_text": "code with <ID_1>...",
  "target_text": "{\"<ID_1>\": \"id\"}"
}
```

LLM should output JSON:
```json
{"<ID_1>": "variableName", "<ID_2>": "anotherName"}
```

