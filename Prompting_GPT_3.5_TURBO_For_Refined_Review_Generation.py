#Installing necessary libraries
!pip install openai pandas openpyxl

# Import required libraries
import openai
import pandas as pd

# Reading FILE for groundtruth generation
df=pd.read_excel("Path to save file/Input_file.xlsx",engine='openpyxl')

#Initializing OPEN_API_KEY
openai.api_key = '<YOUR_API_KEY>'

# Initialize a new column 'Refined_Review_using_Prompt' in the DataFrame
df['Refined_Review_using_Prompt']=""

# Generating Refined Reviews (groundtruth) for whole file by prompting gpt-3.5-turbo via API
for i in range(len(df)):
    App Review = {df['Extracted_Review'][i]} 
    message= f'''<YOUR PROMPT with App Review>'''
    # Sending request to gpt-3.5-turbo
    model="gpt-3.5-turbo"
    messages = [{"role": "user", "content": message}]
    response = openai.chat.completions.create(
                model=model,
                messages=messages,  
            )
    df['Refined_Review_using_Prompt'][i]=response.choices[0].message.content

# Save the refined reviews generated using prompts to a output excel file
df.to_excel("Path to save file/Output_file.xlsx",index=False)
