import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ---------------------------
# SETTINGS — change these if needed
# ---------------------------
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
DATASET_PATH = "tokenized_dataset"   # <-- this should be the folder produced by tokenize_dataset.py
OUTPUT_DIR = "phi3-lora"
NUM_EPOCHS = 3
PER_DEVICE_BATCH = 1
GRADIENT_ACCUMULATION = 8
# ---------------------------

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes (QLoRA) config
'''
load_in_4bit: quantization precision
bnb_4bit_use_double_quant: reduces memory usage by quantizing the quantization constants (saves 0.4 bits per param)
bnb_4bit_quant_type: 4-bit float, memory compression, optimized for data that follows a normal distribution
bnb_4bit_compute_dtype: compute datatype, set precision to something that benefits memory as well as your gpu tensor core specialization
'''
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# Load base model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Prepare for k-bit training and add LoRA
model = prepare_model_for_kbit_training(model)

'''
r: rank = number of params in adaption layer, ie. precision
lora_alpha: scale multiplier to weight changes = alpha / rank
target_modules: (q)uery, (k)ey, (v)alue, (o)utput projections are the attention layers
lora_dropout: percentage randomly zeroes parameters to prevent overfitting
'''
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Load processed dataset (expects a DatasetDict with 'train' and 'test')
dataset = load_from_disk(DATASET_PATH)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Data collator for causal LM (creates labels=input_ids)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training args
'''
logging_steps: log every N steps
eval_strategy/save_strategy: eval/save the model after each X
optim: optimizer
gradient_checkpointing: store checkpoints of calculated gradients instead of storing after each forward pass
save_total_limit: save only last N checkpoints
report_to: log to external services?
'''
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    learning_rate=2e-4,
    num_train_epochs=NUM_EPOCHS,
    fp16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    save_total_limit=2,
    report_to="none",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train & save adapter
trainer.train()
model.save_pretrained(f"{OUTPUT_DIR}-adapter")
print(f"✅ LoRA adapter saved to: {OUTPUT_DIR}-adapter")
