# Install necessary packages
pip install --upgrade pip
pip install pandas numpy==1.20.3 --upgrade transformers --upgrade typing-extensions openpyxl jupyter ipywidgets sentencepiece torch datasets nltk --upgrade accelerate bitsandbytes==0.38.0
pip uninstall -y apex


# Import required libraries
import datasets
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import nltk
from transformers import AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer

# Read data from an Excel file
df1= pd.read_excel("/Path_to_file/Training_file.xlsx")
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
df_d = datasets.DatasetDict({"train":train_dataset})

# Define the pre-trained model to use
model_checkpoint = "facebook/bart-large"


# Create a tokenizer using the specified pre-trained model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# Define a preprocessing function for the data
def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["Extracted_Review"],
        padding="max_length",
        return_tensors='np',
        truncation=True
    )
    labels = tokenizer(
        examples["Refined_Review"],
        padding='max_length',
        return_tensors='np',
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize the datasets using the preprocessing function
tokenized_datasets = df_d.map(preprocess_function, batched=True)
nltk.download("punkt")

# Create the Seq2Seq model for App Reviews Refinement
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Define training hyperparameters
batch_size = 1
num_train_epochs = 1
learning_rate = 3e-5 
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
# Set up training arguments
args = Seq2SeqTrainingArguments(
    output_dir=f"Finetuned_model_location",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    logging_steps=logging_steps,
)

# Create the data collator for the model
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
tokenized_datasets = tokenized_datasets.remove_columns(
    df_d["train"].column_names
)

# Create the Seq2SeqTrainer for training the model
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
trainer.train()
# Train the model
