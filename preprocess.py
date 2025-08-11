import os
import re
import nltk
import string
from nltk.corpus import stopwords

NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(NLTK_DATA_PATH)

def download_nltk_resources():
    nltk.download("punkt", download_dir=NLTK_DATA_PATH)
    nltk.download("stopwords", download_dir=NLTK_DATA_PATH)

def load_stopwords(language):
    try:
        return set(stopwords.words(language))
    except OSError:
        print(f"Stopwords not available for '{language}', using English as fallback.")
        return set(stopwords.words("english"))

def clean_text(text, stop_words):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and w.isalpha()]
    return " ".join(tokens)
