# Installing necessary libraries
!pip install textstat evaluate sacrebleu sacremoses bert_score -U spacy -m spacy download en_core_web_lg


# Importing necessary libraries
import pandas as pd
from evaluate import load

# Reading output file contained both raw and refined reviews
df=pd.read_excel('/Path_to_file/Final_refined_review_file.xlsx')

# Calculating SARI score using Extracted, Ground_truth, and Refined Reviews
sari = load("sari")
def calculate_sari_score(source, prediction, reference):
    sari_score = sari.compute(sources=[source], predictions=[prediction], references=[[reference]])
    return sari_score

SARI_Score=[]
for i in range(0, len(df["Extracted_Review"])):
    extracted_review = df["Extracted_Review"][i]
    reference_review = df["Ground_Truth"][i]
    refined_review = df["Refined_Review"][i]
    sari_score = calculate_sari_score(extracted_review, refined_review, refernce_review)
    SARI_Score.append(sari_score["sari"])

# Calculate the average score
Average_SARI_score = sum(SARI_Score) / len(df["Extracted_Review"]
print(Average_SARI_score)



# Calculating BERTScore Precision using Ground_truth, and Refined Reviews
bertscore = load("bertscore")
def calculate_bert_score(prediction, reference):
    bert_score = bertscore.compute(predictions=[prediction], references=[[reference]], lang="en")
    return bert_score

BP_score=[]
for i in range(0, len(df1["Refined_Review"])):
    refined_review = df1["Refined_Review"][i]
    reference_review = df["Ground_Truth"][i]
    scores = calculate_bert_score(refined_review, reference_review)
    BP_score.append(scores["precision"])

# Calculate the average score
Average_BP_score = sum(([x[0] for x in BP_score])) / len(df["Refined_Review"]
print(Average_BP_score)