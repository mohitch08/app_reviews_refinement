# app_reviews_refinement


# This work has been published in EMNLP 2024(Empirical Methods in Natural Language Processing) conference in industrial track

 #This directory contains the Benchmark RARE Dataset and Code File as described in the paper:
- RARE_Dataset: In this folder, we introduce RARE, a benchmark for App Review Refinement. This folder contains two subfolders named Gold_Corpus and Silver_Corpus.

1. Gold_Corpus: In this folder, a corpus of 10,000 annotated reviews, collaboratively refined by software engineers and a large language model (LLM) sourced from 10 different application domains, is provided.

2. Silver_Corpus: This folder includes a set of 10,000 automatically refined reviews using the best-performing model, Flan-T5, which was trained on 10,000 reviews from the gold corpus, forming the silver corpus.

- Code_File: In this folder, all the code files used in the entire experiment and research are provided. This folder contains four subfolders named Data_Extraction,      Refined_Review_Generation_through_Prompting, Model_Finetuning_and_Inferences, and Result_Evaluation.

1. Data_Extraction: This folder contains 2 Python files named 'Google_Play_Store_Reviews_Extraction_from_10_different_App.py', which was used for extracting 10,000 raw reviews from the Google Play Store, and 'Apple_App_Store_Reviews_Extraction_from_10_different_App.py', which was used for extracting 10,000 raw reviews from the Apple App Store.

2. Refined_Review_Generation_through_Prompting: This folder contain a Python file named 'Prompting_GPT_3.5_TURBO_For_Refined_Review_Generation.py', which was used to guide GPT-3.5-Turbo in generating refined versions of the raw reviews.

3. Model_Finetuning_and_Inferences: This folder contains 16 Python files: one for fine-tuning and another for inference, each for eight models, including BART, Flan-T5, Pegasus, Llama-2, Falcon, Mistral, Orca-2, and Gemma.

4. Result_Evaluation: This folder contains 2 Python files: 'Reference_free_Automatic_Metrics_Evaluation.py' for evaluating reference-free metrics such as FKGL, FKRE, LEN, and SS, and 'Reference_Based_Automatic_Metrics_Evaluation.py' for evaluating reference-based metrics such as SARI and BERTScore Precision
