import json
from datasets import load_dataset
from transformers import AutoTokenizer

# --------- SETTINGS ----------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
INPUT_JSON = "chatlogs_combined.json"
OUTPUT_DATASET = "tokenized_dataset"
MAX_LEN = 256
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(example):
    text = f"{example['prompt']} {example['response']}"
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )

# Load JSON
dataset = load_dataset("json", data_files=INPUT_JSON)

# Split dataset into 90% train and 10% validation, seed=42 for reproducability (arbitrary)
split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

# Tokenize every data point
tokenized = split_dataset.map(tokenize, batched=False)

# Save processed dataset
tokenized.save_to_disk(OUTPUT_DATASET)

print(f"✅ Dataset processed & saved to: {OUTPUT_DATASET}")
