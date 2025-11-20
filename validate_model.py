#!/usr/bin/env python
"""
Validation script for the trained variable name repair model.
Loads saved model weights and validates on 1000 examples.
"""
import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import tinker
from tinker import types
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch

# =========================
# Hard coded config
# =========================

VALIDATION_PATH = "data/data_cpp.jsonl"  # JSONL file path
BASE_MODEL = "meta-llama/Llama-3.1-8B"
CHECKPOINT_URL = "tinker://ca272a9b-69f0-500f-b444-a806805acac9:train:0/sampler_weights/proj-final"
NUM_VALIDATION_EXAMPLES = 200
VALIDATION_OFFSET = 300  # Start from this index in the dataset
MAX_SEQ_LENGTH = 8192    # Max tokens per example (should match training)
SEED = 42
MAX_GENERATION_TOKENS = 100  # Enough for 5 names
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model

# Global variable for embedding model (initialized in main)
_embedding_model = None
_embedding_tokenizer = None

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
    Build the prompt for a single example, asking for 5 ranked suggestions.

    mask_token is something like "<ID_1>".
    """
    prompt = (
        "You are a coding assistant that suggests clear and descriptive variable names.\n\n"
        "Here is a function with one placeholder variable name.\n\n"
        f"Placeholder token: {mask_token}\n\n"
        "<code>\n"
        f"{code}\n"
        "</code>\n\n"
        f"Suggest 5 possible variable names for {mask_token}, ranked from best to worst.\n"
        "Provide only the names, one per line:\n"
        "1."
    )
    return prompt


def get_embedding(text: str) -> np.ndarray:
    """
    Get embedding vector for a text string using the global embedding model.
    """
    global _embedding_model, _embedding_tokenizer
    
    # Tokenize and get embeddings
    inputs = _embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    with torch.no_grad():
        outputs = _embedding_model(**inputs)
        # Use mean pooling of token embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1)
    
    return embeddings.squeeze().numpy()


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1, where 1 is identical direction.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return float(dot_product / (norm1 * norm2))


def calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate similarity score between two strings using embedding-based cosine similarity.
    Returns a value between 0 and 1, where 1 is most similar.
    """
    # Handle exact match quickly
    if str1.lower() == str2.lower():
        return 1.0
    
    # Handle empty strings
    if not str1 or not str2:
        return 0.0
    
    # Get embeddings for both strings
    emb1 = get_embedding(str1)
    emb2 = get_embedding(str2)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(emb1, emb2)
    
    # Normalize to 0-1 range (cosine similarity is -1 to 1)
    # For variable names, we expect positive similarity, but clip to be safe
    normalized_similarity = (similarity + 1) / 2
    
    return max(0.0, min(1.0, normalized_similarity))


def parse_top_5_names(response: str) -> List[str]:
    """
    Parse the top 5 names from the model's response.
    Expected format:
    1. name1
    2. name2
    3. name3
    4. name4
    5. name5
    
    Returns a list of up to 5 names (may be fewer if parsing fails).
    """
    lines = response.strip().split('\n')
    names = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try to parse numbered format like "1. name" or "1) name" or "1 name"
        # Also handle cases where the number/prefix is missing
        parts = line.split('.', 1)
        if len(parts) == 2:
            name = parts[1].strip()
        else:
            parts = line.split(')', 1)
            if len(parts) == 2:
                name = parts[1].strip()
            else:
                # No number prefix, take the whole line
                name = line.strip()
        
        # Remove any remaining numbering or special characters at start
        while name and name[0] in '0123456789.)- ':
            name = name[1:]
        
        name = name.strip()
        if name:
            names.append(name)
        
        if len(names) >= 5:
            break
    
    return names[:5]


def predict_top_5_names_baseline(
    model,
    tokenizer,
    code_snippet: str,
    mask_token: str,
    max_tokens: int = MAX_GENERATION_TOKENS,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> List[str]:
    """
    Predict top 5 variable names using HuggingFace model directly (for baseline).
    Returns a list of up to 5 names.
    """
    prompt = build_prompt(code_snippet, mask_token)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.0001,  # Near-zero for greedy (can't be exactly 0)
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode only the generated part (skip the prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Parse the response to extract the 5 names
    names = parse_top_5_names(decoded)
    
    return names


def predict_top_5_names(
    sampling_client,
    tokenizer,
    code_snippet: str,
    mask_token: str,
    max_tokens: int = MAX_GENERATION_TOKENS
) -> List[str]:
    """
    Predict top 5 variable names given code and mask token using Tinker client.
    Returns a list of up to 5 names.
    """
    prompt = build_prompt(code_snippet, mask_token)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,  # greedy decoding for validation
        stop=[],  # Don't stop early, let it generate all 5
    )
    future = sampling_client.sample(
        prompt=model_input,
        sampling_params=params,
        num_samples=1,
    )
    result = future.result()
    decoded = tokenizer.decode(result.sequences[0].tokens, skip_special_tokens=True)
    
    # Parse the response to extract the 5 names
    names = parse_top_5_names(decoded)
    
    return names


def validate(
    sampling_client,
    tokenizer,
    validation_data: List[Dict],
    max_seq_length: int = MAX_SEQ_LENGTH,
    baseline_mode: bool = False,
    baseline_model=None
) -> Dict:
    """
    Run validation on all examples and compute metrics.
    Computes three types of scores:
    - Exact score: 1 if top prediction matches exactly, 0 otherwise
    - Top-5 score: 1 if correct name is in top 5, 0 otherwise
    - Partial score: Similarity score (0-1) between top prediction and correct name
    
    Skips examples that exceed max_seq_length.

    Args:
        sampling_client: Tinker sampling client (if not baseline_mode)
        tokenizer: Tokenizer for the model
        validation_data: List of validation examples
        max_seq_length: Maximum sequence length
        baseline_mode: If True, use HuggingFace model directly
        baseline_model: HuggingFace model (required if baseline_mode=True)

    Returns:
        Dictionary with validation metrics including checkpoints every 50 samples
    """
    total = len(validation_data)
    exact_score_sum = 0
    top5_score_sum = 0
    partial_score_sum = 0
    predictions = []
    skipped = 0
    evaluated = 0
    
    # Track metrics at each 50-sample checkpoint
    checkpoints = []

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

        # Get top 5 predictions (different method for baseline vs fine-tuned)
        if baseline_mode:
            pred_names = predict_top_5_names_baseline(baseline_model, tokenizer, code, mask_token)
        else:
            pred_names = predict_top_5_names(sampling_client, tokenizer, code, mask_token)
        
        # Default to empty list if no predictions
        if not pred_names:
            pred_names = [""]
        
        top_pred = pred_names[0]
        
        # Calculate exact score (1 if exact match, 0 otherwise)
        exact_score = 1 if top_pred == target_name else 0
        exact_score_sum += exact_score
        
        # Calculate top-5 score (1 if target in top 5, 0 otherwise)
        top5_score = 1 if target_name in pred_names else 0
        top5_score_sum += top5_score
        
        # Calculate partial score (similarity between top prediction and target)
        partial_score = calculate_similarity(top_pred, target_name)
        partial_score_sum += partial_score

        predictions.append({
            "file": example["file"],
            "mask_token": mask_token,
            "target": target_name,
            "top_5_predictions": pred_names,
            "top_prediction": top_pred,
            "exact_score": exact_score,
            "top5_score": top5_score,
            "partial_score": partial_score
        })

        # Save checkpoint and print progress every 50 examples
        if evaluated % 50 == 0:
            current_exact = exact_score_sum / evaluated * 100
            current_top5 = top5_score_sum / evaluated * 100
            current_partial = partial_score_sum / evaluated * 100
            
            checkpoint = {
                "samples": evaluated,
                "skipped": skipped,
                "exact_matches": int(exact_score_sum),
                "exact_accuracy": current_exact,
                "top5_matches": int(top5_score_sum),
                "top5_accuracy": current_top5,
                "partial_score_avg": current_partial
            }
            checkpoints.append(checkpoint)
            
            print(f"[{evaluated} evaluated, {skipped} skipped] Exact: {current_exact:.2f}% | Top-5: {current_top5:.2f}% | Partial: {current_partial:.2f}%")

    if skipped > 0:
        print(f"\nSkipped {skipped} examples that exceeded {max_seq_length} tokens")

    exact_accuracy = exact_score_sum / evaluated * 100 if evaluated > 0 else 0.0
    top5_accuracy = top5_score_sum / evaluated * 100 if evaluated > 0 else 0.0
    partial_avg = partial_score_sum / evaluated * 100 if evaluated > 0 else 0.0

    metrics = {
        "total_examples": total,
        "evaluated": evaluated,
        "skipped": skipped,
        "exact_matches": int(exact_score_sum),
        "top5_matches": int(top5_score_sum),
        "exact_accuracy": exact_accuracy,
        "top5_accuracy": top5_accuracy,
        "partial_score_avg": partial_avg,
        "checkpoints_every_50": checkpoints,
        "predictions": predictions
    }

    return metrics


def print_summary(metrics: Dict):
    """
    Print a summary of validation results with all three scoring metrics.
    """
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total examples:     {metrics['total_examples']}")
    print(f"Evaluated:          {metrics['evaluated']}")
    print(f"Skipped (too long): {metrics['skipped']}")
    print()
    print(f"Exact matches:      {metrics['exact_matches']} ({metrics['exact_accuracy']:.2f}%)")
    print(f"Top-5 matches:      {metrics['top5_matches']} ({metrics['top5_accuracy']:.2f}%)")
    print(f"Partial score avg:  {metrics['partial_score_avg']:.2f}%")
    print("=" * 70)

    # Show some error examples (exact score = 0)
    predictions = metrics['predictions']
    errors = [p for p in predictions if p['exact_score'] == 0]

    if errors:
        print(f"\nShowing first 10 errors out of {len(errors)} total exact mismatches:")
        for idx, error in enumerate(errors[:10]):
            print(f"\nError {idx + 1}:")
            print(f"  File: {error['file']}")
            print(f"  Mask: {error['mask_token']}")
            print(f"  Target: {error['target']}")
            print(f"  Top prediction: {error['top_prediction']}")
            print(f"  All predictions: {error['top_5_predictions']}")
            print(f"  Partial score: {error['partial_score']:.3f}")
            print(f"  In top-5: {'Yes' if error['top5_score'] == 1 else 'No'}")


def save_results(metrics: Dict, output_path: str = "validation_results.json"):
    """
    Save validation results to a JSON file with checkpoint information.
    """
    # Create a structured output
    output = {
        "summary": {
            "total_examples": metrics["total_examples"],
            "evaluated": metrics["evaluated"],
            "skipped": metrics["skipped"],
            "exact_matches": metrics["exact_matches"],
            "exact_accuracy": metrics["exact_accuracy"],
            "top5_matches": metrics["top5_matches"],
            "top5_accuracy": metrics["top5_accuracy"],
            "partial_score_avg": metrics["partial_score_avg"]
        },
        "checkpoints_every_50_samples": metrics["checkpoints_every_50"],
        "detailed_predictions": metrics["predictions"]
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")
    
    # Also save a compact version with just checkpoints
    checkpoint_path = output_path.replace(".json", "_checkpoints.json")
    checkpoint_output = {
        "summary": output["summary"],
        "checkpoints_every_50_samples": output["checkpoints_every_50_samples"]
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_output, f, indent=2)
    print(f"Checkpoint summary saved to: {checkpoint_path}")


# =========================
# Main
# =========================

def main():
    global _embedding_model, _embedding_tokenizer
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Validate variable name repair model (fine-tuned or baseline)"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use baseline untrained Llama 3.1 model instead of fine-tuned checkpoint"
    )
    args = parser.parse_args()
    
    set_seed(SEED)

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set TINKER_API_KEY in your environment before running this script."
        )

    print("=" * 60)
    print("MODEL VALIDATION SCRIPT")
    print("=" * 60)
    print(f"Base model: {BASE_MODEL}")
    if args.baseline:
        print("Mode: BASELINE (untrained model)")
    else:
        print(f"Mode: FINE-TUNED")
        print(f"Checkpoint: {CHECKPOINT_URL}")
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

    # Initialize embedding model for similarity calculations
    print(f"\nLoading embedding model ({EMBEDDING_MODEL_NAME}) for similarity scoring...")
    _embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    _embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    _embedding_model.eval()  # Set to evaluation mode
    print("Embedding model loaded")

    # Load model and tokenizer
    print("Getting tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        use_fast=False,
    )
    # Common for Llama models: set pad token to eos if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sampling_client = None
    baseline_model = None

    if args.baseline:
        # Use baseline untrained model with HuggingFace Transformers
        print(f"\nLoading baseline model from HuggingFace: {BASE_MODEL}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        baseline_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        if device == "cpu":
            baseline_model = baseline_model.to(device)
        baseline_model.eval()
        print("Baseline model loaded successfully")
        
        # Run validation in baseline mode
        metrics = validate(
            sampling_client=None,
            tokenizer=tokenizer,
            validation_data=validation_data,
            baseline_mode=True,
            baseline_model=baseline_model
        )
    else:
        # Load the fine-tuned checkpoint via Tinker
        print("\nCreating Tinker service client...")
        service_client = tinker.ServiceClient()
        
        print(f"Loading checkpoint from: {CHECKPOINT_URL}")
        sampling_client = service_client.create_sampling_client(
            model_path=CHECKPOINT_URL
        )
        print("Fine-tuned model loaded successfully")
        
        # Run validation with Tinker
        metrics = validate(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            validation_data=validation_data,
            baseline_mode=False,
            baseline_model=None
        )

    # Print summary
    print_summary(metrics)

    # Save results with appropriate filename
    if args.baseline:
        output_file = "validation_results_baseline.json"
    else:
        output_file = "validation_results.json"
    save_results(metrics, output_path=output_file)


if __name__ == "__main__":
    main()
