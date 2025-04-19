# Install necessary packages
# pip install --upgrade transformers peft sentencepiece datasets loralib bitsandbytes -q --upgrade pandas openpyxl accelerate ipywidgets -q trl==0.4.7 


# Import required libraries
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, PeftConfig
import torch
import pandas as pd
compute_dtype = getattr(torch, "float16")

# Quantization config : loading model in 4 bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)

# Peft model path
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

# Loading tokenizer and creating its object
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Loading peft finetuned model
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
df= pd.read_excel(""/Path_to_file/Testing_file.xlsx"", engine='openpyxl')

df["Model_output"]=""

# Function to generate inferences for test data
for i in range(len(df)):
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
