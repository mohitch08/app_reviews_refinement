# Install necessary packages
pip uninstall -y wheel typing-extensions tqdm python-dateutil torch pandas torchvision h5py apex
pip install h5py==3.10.0 typing-extensions==4.9.0 wheel==0.42.0 accelerate==0.25.0 bitsandbytes==0.40.2 cmake==3.28.1 datasets==2.16.0 dill==0.3.7 einops==0.7.0 et-xmlfile==1.1.0 fsspec==2023.10.0 huggingface-hub==0.20.1 lit==17.0.6 mpmath==1.3.0 multiprocess==0.70.15 openpyxl==3.1.2 pandas==2.0.3 peft==0.4.0 pillow==10.1.0 pyarrow==14.0.2 pyarrow-hotfix==0.6 python-dateutil==2.8.2 safetensors==0.4.1 sympy==1.12 tokenizers==0.13.3 torch==2.0.0 torchvision==0.15.1 tqdm==4.66.1 transformers==4.31.0 triton==2.0.0 trl==0.4.7 tzdata==2023.3 xlrd==2.0.1 xxhash==3.4.1

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
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# Read data from an Excel file
df= pd.read_excel("/Path_to_file/Training_file.xlsx")

# Split the data into input (X) and output (y) columns
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Create a new DataFrame with relevant columns for the training dataset
train_df = pd.DataFrame({
    "Extracted_Review" : X['Extracted_Review'],
    "Refined_Review" : y,
})

# Convert the DataFrame to a Hugging Face datasets.Dataset
train_dataset = datasets.Dataset.from_dict(train_df)


# Define the pre-trained model to use
model_checkpoint = "tiiuae/falcon-7b-instruct"

# Create a tokenizer using the specified pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Define a preprocessing function for the data
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['Extracted_Review'])):
        text = f"### Input: {example['Extracted_Review'][i]}\n ### Output: {example['Refined_Review'][i]}"
        output_texts.append(text)
    return output_texts

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)
# Create the model for App Reviews Refinement
model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True
)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value"
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ],
)
model.config.use_cache = False
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Training Arguments
batch_size = 4
num_train_epochs = 8
learning_rate = 2e-4
# Show the training loss with every epoch
logging_steps = len(train_dataset["Extracted_Review"]) // 4
model_name = model_checkpoint.split("/")[-1]
# Define training hyperparameters
training_arguments = TrainingArguments(
    output_dir=f"Finetuned_model_location",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_steps=500,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=0.001,
    fp16=True,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)
model.config.use_cache = False

# Create the SFTTTrainer for training the model
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Iterate through the named modules of the trainer's model
for name, module in trainer.model.named_modules():
    # Check if the name contains "norm"
    if "norm" in name:
        module = module.to(torch.float32)

trainer.train()
# Train the model
