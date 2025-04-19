# Install necessary packages
pip install --upgrade pip
pip install --upgrade transformers pandas
pip install peft trl==0.4.7 huggingface_hub openpyxl ipywidgets sentencepiece datasets loralib bitsandbytes transformers accelerate ipywidgets -q


# Import required libraries
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer,AutoModelForSeq2SeqLM
from peft import LoraConfig, PeftModel, LoraModel, PeftConfig
import torch
import pandas as pd
from huggingface_hub import login
compute_dtype = getattr(torch, "float16")

# Quantization configuration
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)

# Token id to access gemma model
hf_token = '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'
login(token = hf_token)

# Peft model id
peft_model_id="Finetuned_model_location"
config = PeftConfig.from_pretrained(peft_model_id)

# Model
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=quant_config,
    device_map={"": 0},
    trust_remote_code=True
)

# Tokenizer 
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, peft_model_id)

# Output generation parameters
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config_temperature = 1
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config_eod_token_id = tokenizer.eos_token_id

# Read test data from excel file
df= pd.read_excel("/Path_to_file/Testing_file.xlsx", engine='openpyxl')
df["Model_output"]=""

# Function to generate inferences for test data
for i in range(len(df):
    query = "### Input:" + df['Extracted_Review'][i] + "\n### Output:"
    encoding = tokenizer(query, return_tensors="pt").to("cuda:0")
    outputs = model.generate(input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = out.split('Output:')[1].strip()
  
    df["Model_output"][i]=cleaned
    # Save inferences as excel file
    df.to_excel("/Path_to_file/Final_refined_review_file.xlsx",index=False)
