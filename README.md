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

Output will be saved to `data/data_cpp.jsonl`. 
