#!/usr/bin/env python
"""
Validation script for the trained variable name repair model.
Loads saved model weights and validates on 1000 examples.
"""
import json
import os
import random
from typing import Dict, List

import numpy as np
import tinker
from tinker import types

# =========================
# Hard coded config
# =========================

VALIDATION_PATH = "data/data_cpp.jsonl"  # JSONL file path
BASE_MODEL = "meta-llama/Llama-3.1-8B"
MODEL_NAME = "cpp-var-name-model"  # Name of the saved model to load
NUM_VALIDATION_EXAMPLES = 1000
VALIDATION_OFFSET = 0  # Start from this index in the dataset
MAX_SEQ_LENGTH = 8192    # Max tokens per example (should match training)
SEED = 42
MAX_GENERATION_TOKENS = 16

# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl_cpp(path: str, max_examples=None, offset=0):
    """
    Load data_cpp.jsonl with lines like:
    {
      "file": "row_000001",
      "func_name": "id",
      "input_text": "IfcComplexPropertyTemplate(... <ID_1> ...)",
      "target_text": "{\"<ID_1>\": \"id\"}"
    }

    Returns a list of dicts with:
      code:        original input_text
      mask_token:  placeholder token (e.g. "<ID_1>")
      target_name: variable name to predict (e.g. "id")
    """
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            # Skip to offset
            if idx < offset:
                continue
            
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            input_text = raw["input_text"]
            target_text = raw["target_text"]

            # target_text itself is a JSON string like "{\"<ID_1>\": \"id\"}"
            mapping = json.loads(target_text)
            if len(mapping) != 1:
                # Skip examples with multiple targets for now
                continue

            mask_token, target_name = next(iter(mapping.items()))

            example = {
                "file": raw.get("file", "unknown"),
                "code": input_text,
                "mask_token": mask_token,
                "target_name": target_name,
            }
            data.append(example)
            if max_examples is not None and len(data) >= max_examples:
                break
    return data


def build_prompt(code: str, mask_token: str) -> str:
    """
    Build the prompt for a single example.
    Must match the training prompt format exactly.
    
    mask_token is something like "<ID_1>".
    """
    prompt = (
        "You are a coding assistant that suggests clear and descriptive variable names.\n\n"
        "Here is a function with one placeholder variable name.\n\n"
        f"Placeholder token: {mask_token}\n\n"
        "<code>\n"
        f"{code}\n"
        "</code>\n\n"
        f"The placeholder {mask_token} should be renamed to:"
    )
    return prompt


def predict_name(
    sampling_client,
    tokenizer,
    code_snippet: str,
    mask_token: str,
    max_tokens: int = MAX_GENERATION_TOKENS
) -> str:
    """
    Predict a variable name given code and mask token.
    """
    prompt = build_prompt(code_snippet, mask_token)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # greedy decoding for validation
        stop=["\n"],
    )
    future = sampling_client.sample(
        prompt=model_input,
        sampling_params=params,
        num_samples=1,
    )
    result = future.result()
    decoded = tokenizer.decode(result.sequences[0].tokens)
    return decoded.strip()


def validate(
    sampling_client,
    tokenizer,
    validation_data: List[Dict],
    max_seq_length: int = MAX_SEQ_LENGTH
) -> Dict:
    """
    Run validation on all examples and compute metrics.
    Skips examples that exceed max_seq_length.
    
    Returns:
        Dictionary with validation metrics
    """
    total = len(validation_data)
    exact_matches = 0
    predictions = []
    skipped = 0
    evaluated = 0
    
    print(f"\nRunning validation on {total} examples...")
    print("=" * 60)
    
    for idx, example in enumerate(validation_data):
        code = example["code"]
        mask_token = example["mask_token"]
        target_name = example["target_name"]
        
        # Check sequence length before prediction
        prompt = build_prompt(code, mask_token)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        
        if len(prompt_tokens) > max_seq_length:
            skipped += 1
            continue
        
        evaluated += 1
        
        # Get prediction
        pred_name = predict_name(sampling_client, tokenizer, code, mask_token)
        
        # Check for exact match
        is_correct = (pred_name == target_name)
        if is_correct:
            exact_matches += 1
        
        predictions.append({
            "file": example["file"],
            "mask_token": mask_token,
            "target": target_name,
            "prediction": pred_name,
            "correct": is_correct
        })
        
        # Print progress
        if evaluated % 50 == 0 or evaluated == 1:
            current_acc = exact_matches / evaluated * 100
            print(f"[{evaluated} evaluated, {skipped} skipped] Accuracy so far: {current_acc:.2f}%")
        
        # Print some example predictions
        if evaluated <= 5 or (not is_correct and evaluated <= 20):
            status = "✓" if is_correct else "✗"
            print(f"\n{status} Example {evaluated}:")
            print(f"  Mask: {mask_token}")
            print(f"  Target: {target_name}")
            print(f"  Prediction: {pred_name}")
    
    if skipped > 0:
        print(f"\nSkipped {skipped} examples that exceeded {max_seq_length} tokens")
    
    accuracy = exact_matches / evaluated * 100 if evaluated > 0 else 0.0
    
    metrics = {
        "total_examples": total,
        "evaluated": evaluated,
        "skipped": skipped,
        "exact_matches": exact_matches,
        "accuracy": accuracy,
        "predictions": predictions
    }
    
    return metrics


def print_summary(metrics: Dict):
    """
    Print a summary of validation results.
    """
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total examples:     {metrics['total_examples']}")
    print(f"Evaluated:          {metrics['evaluated']}")
    print(f"Skipped (too long): {metrics['skipped']}")
    print(f"Exact matches:      {metrics['exact_matches']}")
    print(f"Accuracy:           {metrics['accuracy']:.2f}%")
    print("=" * 60)
    
    # Show some error examples
    predictions = metrics['predictions']
    errors = [p for p in predictions if not p['correct']]
    
    if errors:
        print(f"\nShowing first 10 errors out of {len(errors)} total errors:")
        for idx, error in enumerate(errors[:10]):
            print(f"\nError {idx + 1}:")
            print(f"  File: {error['file']}")
            print(f"  Mask: {error['mask_token']}")
            print(f"  Target: {error['target']}")
            print(f"  Prediction: {error['prediction']}")


def save_results(metrics: Dict, output_path: str = "validation_results.json"):
    """
    Save validation results to a JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set TINKER_API_KEY in your environment before running this script."
        )

    print("=" * 60)
    print("MODEL VALIDATION SCRIPT")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    print(f"Saved model name: {MODEL_NAME}")
    print(f"Validation examples: {NUM_VALIDATION_EXAMPLES}")
    print("=" * 60)

    # Load validation data
    print(f"\nLoading validation data from {VALIDATION_PATH}...")
    validation_data = load_jsonl_cpp(
        VALIDATION_PATH,
        max_examples=NUM_VALIDATION_EXAMPLES,
        offset=VALIDATION_OFFSET
    )
    print(f"Loaded {len(validation_data)} validation examples")

    # Create Tinker service and load saved model
    print("\nCreating Tinker service client...")
    service_client = tinker.ServiceClient()
    
    print(f"Loading saved model '{MODEL_NAME}' as sampling client...")
    # Load the saved LoRA adapter directly as a sampling client
    sampling_client = service_client.create_lora_sampling_client(
        base_model=BASE_MODEL,
        name=MODEL_NAME
    )
    
    print("Getting tokenizer...")
    tokenizer = sampling_client.get_tokenizer()
    
    # Run validation
    metrics = validate(sampling_client, tokenizer, validation_data)
    
    # Print summary
    print_summary(metrics)
    
    # Save results
    save_results(metrics)


if __name__ == "__main__":
    main()

