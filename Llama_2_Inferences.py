# Installing necessary libraries
!pip install torch peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7 accelerate typing-extensions --upgrade --upgrade torchvision torch==1.13.0 numpy==1.20.3 openpyxl

# Importing necessary libraries
from transformers import BitsAndBytesConfig, AutoTokenizer, LlamaForCausalLM
from peft import LoraConfig, PeftModel, LoraModel, PeftConfig
import torch
import pandas as pd

# Initializing adapter from finetuned model
adapters_name="Finetuned_Model_Location"

# Initializing model name
model_name="NousResearch/Llama-2-7b-chat-hf"

# Quantization config : loading model in 4 bit
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Model
base_model = LlamaForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map={"": 0}
)

# Loading peft finetuned model
M = PeftModel.from_pretrained(base_model, adapters_name)

# Loading tokenizer and creating its object
llama_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
llama_tokenizer.pad_token = llama_tokenizer.eos_token
llama_tokenizer.padding_side = "right"  

# Reading the extracted review test datatset to generate inferences
df= pd.read_excel("/Path_to_file/Testing_file.xlsx", engine='openpyxl')

df["Model_output"]=""

# Running inferences
for i in range(len(df)):
    query = "### Input:"+df['Extracted_Review'][i]+"\n### Output:"
    inputs = llama_tokenizer(query, return_tensors="pt")
    outputs = M.generate(**inputs, do_sample=True, num_beams=1, max_new_tokens=200)
    out=llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    out=str(out)
    out=out.split("###")[2]
    if "Output" in out:
        out=out[9:]
    indexr = out.index('Output')
    cleaned=out[indexr+7:]
    cleaned=cleaned.strip()
    df["Model_output"][i]=cleaned
    df.to_excel(""/Path_to_file/Final_refined_review_file.xlsx",index=False")
