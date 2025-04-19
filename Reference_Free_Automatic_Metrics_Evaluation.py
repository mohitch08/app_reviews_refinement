# Installing necessary libraries
!pip install textstat evaluate sacrebleu sacremoses bert_score -U spacy -m spacy download en_core_web_lg


# Importing necessary libraries
import pandas as pd
import textstat
import spacy

# Reading output file contained both raw and refined reviews
df=pd.read_excel('/Path_to_file/Final_refined_review_file.xlsx')

# Calculating FKGL score for raw and refined reviews
def calculate_fk_grade_level(text):
  return textstat.flesch_kincaid_grade(text)

def calculate_fkgl_for_column(df, column_name):
    score = []
    for i in range(len(df[column_name])):
        refined_review = df[column_name][i]
        FKGL_score = calculate_fk_grade_level(refined_review)
        score.append(FKGL_score)
    return sum(score) / len(df[column_name])

# Assuming df is your DataFrame
columns_to_analyze = ["Extracted_Review", "Refined_Review"]
for column in columns_to_analyze:
    fkgl_score = calculate_fkgl_for_column(df, column)
    print(f"FKGL for {column}: {fkgl_score}")



# Calculating FKRE score for raw and refined reviews
def calculate_fk_reading_ease(text):
  return textstat.flesch_reading_ease(text)

def calculate_fkr_for_column(df, column_name):
    score = []
    for i in range(len(df[column_name])):
        refined_review = df[column_name][i]
        FKRE_score = calculate_fk_reading_ease(refined_review)
        score.append(FKRE_score)
    return sum(score) / len(df[column_name])

# Assuming df is your DataFrame
columns_to_analyze = columns_to_analyze = ["Extracted_Review", "Refined_Review"]
for column in columns_to_analyze:
    fkr_score = calculate_fkr_for_column(df, column)
    print(f"FKRE for {column}: {fkr_score}")



# Calculating LEN score for raw and refined reviews
def calculate_sentence_count(text):
  return textstat.lexicon_count(text, removepunct=True), textstat.sentence_count(text)

def calculate_len_sent_for_column(df, column_name):
    score = []
    for i in range(len(df[column_name])):
        refined_review = df[column_name][i]
        words,sent = calculate_sentence_count(refined_review)
        len_score = words/sent
        score.append(len_score)
    return sum(score) / len(df[column_name])

# Assuming df is your DataFrame
columns_to_analyze = ["Extracted_Review", "Refined_Review"]
for column in columns_to_analyze:
    len_score = calculate_len_sent_for_column(df, column)
    print(f"LEN for {column}: {len_score}")



# Calculating Similarity Score (SS) for raw and refined reviews
# Load spaCy model
nlp = spacy.load("en_core_web_lg")

def calculate_semantic_similarity(text1, text2):
    # Tokenize and process texts with spaCy
    doc1 = nlp(text1)
    doc2 = nlp(text2)

    # Calculate similarity between document vectors
    similarity = doc1.similarity(doc2)
    return similarity

def calculate_word_overlap(text1, text2):
    # Tokenize texts
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())

    # Calculate overlap
    overlap = len(tokens1.intersection(tokens2)) / (len(tokens1) + len(tokens2))
    return overlap

# Columns to analyze
columns_to_analyze = ["Extracted_Review", "Refined_Review"]

# Initialize lists to store scores
semantic_similarity_scores = []
word_overlap_scores = []

for column in columns_to_analyze:
  for i in range(len(df[column])):
    text1 = df["Extracted_Review"].iloc[i]
    text2 = df[column].iloc[i]
    semantic_similarity = calculate_semantic_similarity(text1, text2)
    word_overlap = calculate_word_overlap(text1, text2)

    semantic_similarity_scores.append(semantic_similarity)
    word_overlap_scores.append(word_overlap)
  average_semantic_similarity = sum(semantic_similarity_scores) / len(semantic_similarity_scores)
  average_word_overlap = sum(word_overlap_scores) / len(word_overlap_scores)
  print(f"Column: {column}")
  print(f"Average Semantic Similarity: {average_semantic_similarity}")
  print()

