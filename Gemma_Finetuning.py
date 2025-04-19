# Install necessary packages
pip install --upgrade pip transformers pandas
pip install peft ipywidgets sentencepiece datasets loralib bitsandbytes openpyxl transformers accelerate ipywidgets trl==0.4.7 huggingface_hub
-q

# Import required libraries
import torch
import datasets
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
    )
from peft import LoraConfig
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from huggingface_hub import login


# Read data from an Excel file
df= pd.read_excel("/Path_to_file/Training_file.xlsx", engine='openpyxl')


# Create a new DataFrame with relevant columns for the training dataset
train_df = pd.DataFrame({
    "Extracted_Review" : df['Extracted_Review'],
    "Gold_Standard" : df['Ground_Truth'],
})

# Convert the DataFrame to a Hugging Face datasets
train_dataset = datasets.Dataset.from_dict(train_df)


# Token id to access gemma model 
hf_token = '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'
login(token = hf_token)
base_model_name = "google/gemma-7b"
refined_model = "testPrompt6_gemma_app_reviews" #You can give it your own name

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Prompt function 
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Extracted_Review'])):
        text = f"### Input: {example['Extracted_Review'][i]}\n ### Output: {example['Gold_Standard'][i]}"
        output_texts.append(text)
    return output_texts


# Show the training loss with every epoch
logging_steps = len(train_dataset["Extracted_Review"]) // 4


# Quantization Config
quant_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False
)

# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1
# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=32,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

# Training Parameters
train_params = TrainingArguments(
    output_dir=f"Finetuned_model_location",
    num_train_epochs=4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=logging_steps,
    learning_rate=3e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)


fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    peft_config=peft_parameters,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    args=train_params
)

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
