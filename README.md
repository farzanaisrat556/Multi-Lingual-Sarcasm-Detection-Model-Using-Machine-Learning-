# Multilingual Sarcasm Detection

A text classification pipeline that detects sarcasm across five languages — English, Hindi, Bangla, Urdu, and Arabic — using TF-IDF features and a linear SVM.

## What it does

* **Loads per-language CSV datasets** (`text`, `label`, `language` columns).
* **Cleans and normalizes text per language** (lowercasing, Unicode-safe punctuation stripping, tokenization, stopword removal).
* **De-duplicates each dataset**, then balances all five languages to the size of the smallest one so no single language dominates training.
* **Combines everything into one dataset** and trains a linear-kernel SVM on TF-IDF unigram/bigram features.
* **Reports evaluation metrics**: accuracy, precision, recall, F1, and a confusion matrix.

## Results

On a 5,660-row balanced dataset (1,132 rows/language, 80/20 train/test split):

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.82 |
| **Precision (weighted)** | 0.82 |
| **Recall (weighted)** | 0.82 |
| **F1 (weighted)** | 0.82 |

## Project structure

```text
.
├── sarcasm_pipeline.py    # Main pipeline: preprocessing + training
├── requirements.txt
├── README.md
├── .gitignore
└── data/                  # Not included in the repo — see "Data" below
