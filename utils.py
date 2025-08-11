import os
import pandas as pd
from .preprocess import clean_text, load_stopwords

def load_and_balance_data(data_folder):
    datasets = []
    min_samples = None

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(data_folder, file))
            if "text" not in df.columns or "label" not in df.columns or "language" not in df.columns:
                raise ValueError(f"{file} must contain 'text', 'label', 'language'")
            datasets.append(df)
            if min_samples is None or len(df) < min_samples:
                min_samples = len(df)

    balanced = [df.sample(n=min_samples, random_state=42) for df in datasets]
    return pd.concat(balanced, ignore_index=True)

def preprocess_dataset(df):
    df["stopwords"] = df["language"].apply(load_stopwords)
    df["clean_text"] = [clean_text(t, sw) for t, sw in zip(df["text"], df["stopwords"])]
    return df
