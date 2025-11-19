#!/usr/bin/env python
import json
import os
import random

import numpy as np
import tinker
from tinker import types

# =========================
# Hard coded config
# =========================

TRAIN_PATH = "data/data_cpp.jsonl"  # JSONL file path
BASE_MODEL = "meta-llama/Llama-3.1-8B"
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_STEPS = 1000
PRINT_EVERY = 50
MAX_EXAMPLES = None      # Set to an int if you  want to cap dataset size
MAX_SEQ_LENGTH = 8192    # Max tokens per example (Tinker limit is 32768, use conservative value)
SEED = 42
SAVE_NAME = "cpp-var-name-model"


# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl_cpp(path: str, max_examples=None):
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
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            input_text = raw["input_text"]
            target_text = raw["target_text"]

            # target_text itself is a JSON string like "{\"<ID_1>\": \"id\"}"
            mapping = json.loads(target_text)
            if len(mapping) != 1:
                continue
                """
                raise ValueError(
                    f"Expected exactly one mapping in target_text, got {mapping}"
                )
                """
            mask_token, target_name = next(iter(mapping.items()))

            example = {
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


def process_example(example: dict, tokenizer, max_seq_length: int) -> types.Datum:
    """
    Convert one example into a Tinker Datum.

    We set loss weight:
      0 for all prompt tokens
      1 for the variable name tokens
      
    Returns None if the example exceeds max_seq_length.
    """
    code = example["code"]
    mask_token = example["mask_token"]
    target_name = example["target_name"]

    prompt = build_prompt(code, mask_token)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0] * len(prompt_tokens)

    # Add a leading space so the name is separated from the colon
    completion_text = " " + target_name + "\n"
    completion_tokens = tokenizer.encode(
        completion_text, add_special_tokens=False
    )
    completion_weights = [1] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Check if sequence is too long
    if len(tokens) > max_seq_length:
        return None

    # Next token prediction format
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
        ),
    )


def make_datums(raw_data, tokenizer, max_seq_length):
    """
    Convert raw examples to Tinker Datums, filtering out sequences that are too long.
    """
    datums = []
    filtered_count = 0
    
    for ex in raw_data:
        datum = process_example(ex, tokenizer, max_seq_length)
        if datum is not None:
            datums.append(datum)
        else:
            filtered_count += 1
    
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} examples that exceeded {max_seq_length} tokens")
        print(f"Kept {len(datums)} examples for training")
    
    return datums


def train(training_client, datums, learning_rate, batch_size, num_steps, print_every):
    """
    Simple supervised fine tuning loop with mini batches.
    """
    if len(datums) < batch_size:
        raise ValueError(
            f"Not enough training examples: {len(datums)} < batch_size {batch_size}"
        )

    for step in range(num_steps):
        # Sample a random batch
        batch_indices = random.sample(range(len(datums)), batch_size)
        batch = [datums[i] for i in batch_indices]

        fwdbwd_future = training_client.forward_backward(
            batch, "cross_entropy"
        )
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=learning_rate)
        )

        fwdbwd_result = fwdbwd_future.result()
        _ = optim_future.result()

        if (step + 1) % print_every == 0 or step == 0:
            # Compute weighted loss per token
            logprobs_all = []
            weights_all = []
            for output in fwdbwd_result.loss_fn_outputs:
                # output["logprobs"] is per token
                logprobs_all.append(output["logprobs"].tolist())
            for ex in batch:
                weights_all.append(ex.loss_fn_inputs["weights"].tolist())

            logprobs = np.concatenate(logprobs_all)
            weights = np.concatenate(weights_all)
            # Negative average logprob over positions with weight 1
            loss = -np.dot(logprobs, weights) / max(weights.sum(), 1.0)
            print(f"[step {step + 1}] loss per token: {loss:.4f}")


def build_sampling_client(training_client, tokenizer, save_name):
    """
    Save weights and return a sampling client and a helper function
    that predicts a variable name given code and mask token.
    """
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name=save_name
    )

    def predict_name(code_snippet: str, mask_token: str, max_tokens: int = 16) -> str:
        prompt = build_prompt(code_snippet, mask_token)
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        model_input = types.ModelInput.from_ints(tokens=prompt_tokens)

        params = types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # greedy
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

    return sampling_client, predict_name


# =========================
# Main
# =========================

def main():
    set_seed(SEED)

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError(
            "Please set TINKER_API_KEY in your environment before running this script."
        )

    print("Loading dataset from", TRAIN_PATH)
    raw_data = load_jsonl_cpp(TRAIN_PATH, MAX_EXAMPLES)
    print(f"Loaded {len(raw_data)} examples")

    print("Creating Tinker service and training client...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL
    )

    print("Getting tokenizer...")
    tokenizer = training_client.get_tokenizer()

    print("Converting examples to Tinker datums...")
    datums = make_datums(raw_data, tokenizer, MAX_SEQ_LENGTH)
    print(f"Prepared {len(datums)} datums")
    
    if len(datums) < BATCH_SIZE:
        raise ValueError(
            f"Not enough training examples after filtering: {len(datums)} < batch_size {BATCH_SIZE}"
        )

    print("Starting training...")
    train(
        training_client=training_client,
        datums=datums,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_steps=NUM_STEPS,
        print_every=PRINT_EVERY,
    )

    print("Saving weights and creating sampling client...")
    sampling_client, predict_name = build_sampling_client(
        training_client, tokenizer, SAVE_NAME
    )

    # Quick smoke test on first example
    print("\nQuick test on first training example:")
    test_ex = raw_data[0]
    test_code = test_ex["code"]
    mask_token = test_ex["mask_token"]
    gold_name = test_ex["target_name"]

    pred_name = predict_name(test_code, mask_token)

    print("Code snippet:")
    print(test_code)
    print(f"Mask token: {mask_token}")
    print(f"Gold name: {gold_name}")
    print(f"Predicted name: {pred_name}")


if __name__ == "__main__":
    main()
