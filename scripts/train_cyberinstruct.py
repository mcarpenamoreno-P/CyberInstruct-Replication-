####---------------------- Libraries ----------------------####
import os
import sys
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
####------------------------------------------------------------------####

# Get environment variables for configuration
BASE_MODEL = os.environ.get("BASE_MODEL")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR")
TRAIN_FILE = os.environ.get("TRAIN_FILE")
VAL_FILE = os.environ.get("VAL_FILE")
SEED = int(os.environ.get("SEED", "42"))

for name, val in [
    ("BASE_MODEL", BASE_MODEL),
    ("OUTPUT_DIR", OUTPUT_DIR),
    ("TRAIN_FILE", TRAIN_FILE),
    ("VAL_FILE", VAL_FILE),
]:
    if not val:
        print(f"ERROR: missing variable {name}")
        sys.exit(1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seed for reproducibility across random, numpy, and torch (including CUDA if available).
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"BASE_MODEL: {BASE_MODEL}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print(f"TRAIN_FILE: {TRAIN_FILE}")
print(f"VAL_FILE  : {VAL_FILE}")
print(f"SEED      : {SEED}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

# Configure BitsAndBytes for 4-bit quantization, which allows us to load and fine-tune large language models efficiently on limited hardware.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load tokenizer and model with the specified base model, applying the quantization configuration.
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()

# QLoRA: prepare the model for quantized training.
model = prepare_model_for_kbit_training(model)

# Load and prepare training and validation data.
print("Loading datasets...")
train_dataset = load_dataset("json", data_files=TRAIN_FILE, split="train")
eval_dataset = load_dataset("json", data_files=VAL_FILE, split="train")
print(f"Number of train examples: {len(train_dataset)}")
print(f"Number of val examples  : {len(eval_dataset)}")

# Alpaca template for formatting the training examples.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n{output}"
)

# This function takes the raw examples from the dataset and formats them according to the Alpaca template.
def format_examples(examples):
    return [
        ALPACA_TEMPLATE.format(
            instruction=instr,
            input=inp,
            output=out,
        )
        for instr, inp, out in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
        )
    ]

# Configure the PEFT QLoRA settings.
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-5,
    optim="paged_adamw_8bit",
    num_train_epochs=2,
    lr_scheduler_type="linear",
    warmup_ratio=0.02,
    weight_decay=0.0,
    bf16=True,
    tf32=True,
    max_seq_length=1024,
    logging_steps=20,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    do_eval=True,
    report_to="none",
    dataset_text_field=None,
)

# Initialize the training loop using the SFTTrainer from the TRL library.
print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    formatting_func=format_examples,
    tokenizer=tokenizer,
    args=training_args,
)

print("\nStarting training...")
trainer.train()

# Save the fine-tuned model
print("\nSaving adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Training completed. Adapter saved to {OUTPUT_DIR}")
