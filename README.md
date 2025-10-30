# NeuralVariableNameRepair

Writing clean code is not just about making programs work; it is also about giving variables names that clearly communicate their purpose. This project explores training Llama 3.1 8B to understand and recover missing variable names based on the context of functions in C++.

## Setup

1. **Create `.env` file with your Hugging Face token:**
```bash
echo "HF_TOKEN=hf_your_token_here" > .env
```

2. **Create and activate conda environment:**
```bash
conda env create -f environment.yml
conda activate stack-cpp-extract
```

3. **Run the extraction script:**
```bash
python extract_functions.py
```

Output is saved to `data/data_cpp.jsonl` (sample in `example_output.jsonl`).

## Usage

### Load Data & Evaluate Predictions

```python
from data_pipeline import DataPipeline
from evaluate_predictions import PredictionEvaluator

# Load data
pipeline = DataPipeline('example_output.jsonl')
pipeline.load_data()
input_texts, target_texts = pipeline.get_separate_arrays()

# Assume LLM responses are retrieved, TODO
llm_responses = [llm(text) for text in input_texts]

# Evaluate
evaluator = PredictionEvaluator()
cleaned = evaluator.prepare_llm_responses(llm_responses)
metrics = evaluator.evaluate(cleaned, target_texts)
evaluator.print_report(metrics)
```

### Data Format

Each JSONL entry contains:
- `input_text`: Code with masked variables (`<ID_1>`, `<ID_2>`, etc.)
- `target_text`: JSON mapping of placeholders to actual variable names

LLM should output JSON: `{"<ID_1>": "varName", "<ID_2>": "anotherVar"}`

### Evaluation

Exact string matching: 1 point for perfect match, 0 otherwise. Final accuracy = correct_count / total_trials. 