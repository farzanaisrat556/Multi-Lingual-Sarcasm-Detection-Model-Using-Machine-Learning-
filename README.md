# Multilingual Sarcasm Detection (MULAN)

This project implements a multilingual sarcasm detection system using **Support Vector Machine (SVM)** and **TF-IDF** features to classify text as sarcastic or non-sarcastic across five languages:

- **English**
- **Bangla**
- **Hindi**
- **Urdu**
- **Arabic**

While sarcasm detection in English has been explored extensively, low-resource languages like Bangla and Urdu remain underrepresented. This project addresses that gap by creating a balanced multilingual dataset and evaluating the effectiveness of traditional machine learning methods for sarcasm detection.

---

##  Features
- Handles multilingual text with language-specific preprocessing.
- Balanced dataset of **10,550 samples** (2,110 per language).
- TF-IDF vectorization for feature extraction.
- SVM classifier with hyperparameter tuning via **Grid Search**.
- Performance evaluation with Accuracy, Precision, Recall, and F1-score.
- Visualizations using **Matplotlib**.

---

##  Methodology

### 1. Dataset Preparation
- Five labeled datasets: **English, Bangla, Arabic, Urdu, Hindi**.
- **Preprocessing**:
  - Stopword removal (language-specific).
  - Tokenization.
  - Punctuation removal.
  - Stemming and lemmatization (where applicable).
- Dataset balancing: **2,110 samples per language** (based on smallest dataset).

### 2. Feature Engineering
- **TF-IDF** (Term Frequency–Inverse Document Frequency) for text vectorization.
- Extraction of linguistic features.

### 3. Model Training
- **Support Vector Machine (SVM)** classifier.
- Hyperparameter tuning with **Grid Search**.

### 4. Evaluation Metrics
- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1-score
- Confusion Matrix

---

## 🛠 Tools & Libraries

| Tool           | Function                       | Reason for Selection                    |
|----------------|--------------------------------|------------------------------------------|
| Google Colab   | Model training & evaluation    | Free GPU & cloud runtime                 |
| Python 3.10    | Programming language           | Rich ML ecosystem                        |
| NLTK           | Text preprocessing, tokenization | Comprehensive NLP toolkit               |
| Scikit-learn   | ML implementation              | Robust SVM & TF-IDF support              |
| Matplotlib     | Visualization                  | Easy plotting                            |
| Pandas         | Data handling                  | Fast & efficient data manipulation       |

---

## 📊 Results

| Metric          | Score   |
|-----------------|---------|
| Accuracy        | **85.97%** |
| Precision (W)   | **85.82%** |
| Recall (W)      | **85.97%** |
| F1-score (W)    | **85.87%** |

**Confusion Matrix Breakdown**:
- **True Negatives:** 1288  
- **False Positives:** 129  
- **False Negatives:** 167  
- **True Positives:** 526  

---

##  Analysis
**Strengths**:  
- Balanced performance across all metrics.  
- TF-IDF proved effective for multilingual sarcasm detection.  

**Challenges**:  
- Sarcasm is inherently nuanced; errors occurred mainly in borderline cases.  

**Limitations**:  
- Heavy reliance on TF-IDF, which lacks deep contextual understanding.  
- Dataset size limits model generalization.  

---

##  Future Work
- Incorporate deep learning models like **mBERT** or **XLM-R**.
- Expand dataset size.
- Experiment with **word embeddings** and **transformer-based architectures**.
- Enable **real-time sarcasm detection**.

---

##  Applications
- **Sentiment Analysis** — Improve classification accuracy by distinguishing sarcasm.
- **Social Media Monitoring** — Detect sarcasm in multilingual user posts.
