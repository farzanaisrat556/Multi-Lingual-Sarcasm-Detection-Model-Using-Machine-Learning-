import os
import matplotlib.pyplot as plt
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)
from preprocess import download_nltk_resources
from utils import load_and_balance_data, preprocess_dataset

DATA_FOLDER = "data"
MODEL_PATH = "models/sarcasm_model.joblib"
VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

def train_multilingual_model():
    download_nltk_resources()

    df = load_and_balance_data(DATA_FOLDER)
    df = preprocess_dataset(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, zero_division=0):.4f}")

    ConfusionMatrixDisplay.from_estimator(model, X_test_tfidf, y_test)
    plt.show()

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    dump(model, MODEL_PATH)
    dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

if __name__ == "__main__":
    train_multilingual_model()
