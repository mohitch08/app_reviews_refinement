# Install necessary packages
pip uninstall -y wheel typing-extensions tqdm python-dateutil torch pandas torchvision h5py apex
pip install h5py==3.10.0 typing-extensions==4.9.0 wheel==0.42.0 accelerate==0.25.0 bitsandbytes==0.40.2 cmake==3.28.1 datasets==2.16.0 dill==0.3.7 einops==0.7.0 et-xmlfile==1.1.0 fsspec==2023.10.0 huggingface-hub==0.20.1 lit==17.0.6 mpmath==1.3.0 multiprocess==0.70.15 openpyxl==3.1.2 pandas==2.0.3 peft==0.4.0 pillow==10.1.0 pyarrow==14.0.2 pyarrow-hotfix==0.6 python-dateutil==2.8.2 safetensors==0.4.1 sympy==1.12 tokenizers==0.13.3 torch==2.0.0 torchvision==0.15.1 tqdm==4.66.1 transformers==4.31.0 triton==2.0.0 trl==0.4.7 tzdata==2023.3 xlrd==2.0.1 xxhash==3.4.1

# Import required libraries
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, LoraModel, PeftConfig
import torch
import pandas as pd

# Define quantization and PEFT model configuration
compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True
)
peft_model_id="Finetuned_model_location"
config = PeftConfig.from_pretrained(peft_model_id)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=quant_config,
    device_map={"": 0},
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, peft_model_id)

# Configure generation parameters
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config_temperature = 1
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config_eod_token_id = tokenizer.eos_token_id

# Read input data from excel file
df= pd.read_excel("/Path_to_file/Testing_file.xlsx", engine='openpyxl')
df["Model_output"]=""

# Generate and process refined reviews outputs for each row in the DataFrame
for i in range(len(df)):
    query = "### Input:" + df['Extracted_Review'][i] + "\n### Output:"
    encoding = tokenizer(query, return_tensors="pt").to("cuda:0")
    outputs = model.generate(input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = out.split('Output:')[1].strip()
    df["Model_output"][i]=cleaned
    # Save the refined reviews to a output excel file
    df.to_excel("/Path_to_file/Final_refined_review_file.xlsx",index=False)
