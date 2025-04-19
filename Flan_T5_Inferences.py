# Install necessary packages
pip install --upgrade pip
pip install pandas numpy==1.20.3 --upgrade transformers --upgrade typing-extensions openpyxl jupyter ipywidgets sentencepiece torch datasets nltk --upgrade accelerate bitsandbytes==0.38.0
pip uninstall -y apex

# Import required libraries
import pandas as pd
from transformers import pipeline


# Read data from testing excel file into a pandas DataFrame
df = pd.read_excel("/Path_to_file/Testing_file.xlsx")

# Define the location of the fine-tuned model
hub_model_id = "Finetuned_model_location"

# Create a text refinement pipeline using the fine-tuned model
refinement = pipeline("text2text-generation",model=hub_model_id)

# Initialize a new column 'Refined_Review' in the DataFrame
df['Refined_Review']=""

# Iterate over each row in the DataFrame
for i in range(len(df)):
# Generate a refined review using the refinement pipeline and assign it to the 'Refined_Review' column for the current row
	df['Refined_Review'][i]=refinement(df['Extracted_Review'][i],max_length=400)[0]['generated_text']

# Save the refined reviews to a output excel file
df.to_excel("/Path_to_file/Final_refined_review_file.xlsx",index=False)
