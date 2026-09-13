# Multilingual Sarcasm Detection

A text classification pipeline that detects sarcasm across five languages —
**English, Hindi, Bangla, Urdu, and Arabic** — using TF-IDF features and a
linear SVM.

## What it does

1. Loads per-language CSV datasets (`text`, `label`, `language` columns).
2. Cleans and normalizes text per language (lowercasing, Unicode-safe
   punctuation stripping, tokenization, stopword removal).
3. De-duplicates each dataset, then balances all five languages to the size
   of the smallest one so no single language dominates training.
4. Combines everything into one dataset and trains a linear-kernel SVM on
   TF-IDF unigram/bigram features.
5. Reports accuracy, precision, recall, F1, and a confusion matrix.

## Results

On a 5,660-row balanced dataset (1,132 rows/language, 80/20 train/test
split):

| Metric | Score |
|---|---|
| Accuracy | 0.82 |
| Precision (weighted) | 0.82 |
| Recall (weighted) | 0.82 |
| F1 (weighted) | 0.82 |

## Project structure

```
.
├── sarcasm_pipeline.py    # Main pipeline: preprocessing + training
├── requirements.txt
├── README.md
├── .gitignore
└── data/                  # Not included in the repo — see "Data" below
```

## Setup

```bash
git clone <this-repo-url>
cd <repo-name>
pip install -r requirements.txt
```

The script downloads the NLTK resources it needs (`punkt`, `stopwords`) on
first run into a local `./nltk_data` folder.

## Data

The raw CSV datasets are **not included** in this repo (see `.gitignore`) —
they're either large, third-party, or not mine to redistribute. To run the
pipeline yourself, place five CSVs in a `data/` folder (or anywhere you
like), each with these columns:

| Column | Description |
|---|---|
| `text` | The raw text sample |
| `label` | `0` (not sarcastic) or `1` (sarcastic) |
| `language` | One of: `English`, `Hindi`, `Bangla`, `Urdu`, `Arabic` (case-insensitive) |

## Usage

```bash
python sarcasm_pipeline.py path/to/english.csv path/to/bangla.csv path/to/arabic.csv path/to/urdu.csv path/to/hindi.csv
```

If no paths are given, it falls back to five default filenames in the
current directory (see `default_paths` in `sarcasm_pipeline.py`).

This produces:
- `combined_preprocessed_data.csv` — the cleaned, balanced, combined dataset
- `sarcasm_multilingual_model.joblib` — the trained SVM
- `tfidf_vectorizer_multilingual.joblib` — the fitted vectorizer
- `confusion_matrix.png` — evaluation confusion matrix

## Known limitations

- **Tokenization is English-centric.** NLTK's `word_tokenize` isn't
  script-aware for Devanagari, Bangla, Urdu (RTL), or Arabic (RTL). Text is
  no longer corrupted (see below), but a language-specific tokenizer (e.g.
  `indic-nlp-library` for Hindi/Bangla) would likely improve accuracy
  further.
- **Hard balancing discards data.** Balancing all languages to the smallest
  dataset's size (after de-duplication) means a lot of English/Urdu data is
  unused. Class weighting or oversampling the minority languages is a
  reasonable alternative to explore.
- **Duplicate-heavy source data.** One of the original source files was
  ~93% exact duplicate rows before de-duplication — worth checking data
  collection/scraping methodology if you extend this with more data.

## Fixes applied vs. an earlier draft of this script

This pipeline went through a debugging pass worth documenting for anyone
extending it:

- **Stopword lookup was a silent no-op.** The stopword dictionary was keyed
  by two-letter codes (`en`, `hi`, `bn`, `ur`) plus a mis-cased `Ar`, but the
  datasets' `language` column contains full names (`English`, `Hindi`,
  etc.). Every lookup missed, so stopword removal never actually happened
  in any language. Fixed by keying the dictionary to match the real data
  (`english`, `hindi`, `bangla`, `urdu`, `arabic`).
- **Punctuation stripping corrupted Hindi/Bangla text.** `re.sub(r"[^\w\s]",
  " ", text)` looks Unicode-safe but Python's `\w` doesn't treat combining
  marks (vowel signs/matras) as word characters, so words like `हमारे` were
  shredded into `हम र`. Replaced with a Unicode-category-aware cleaner that
  preserves combining marks.
- **Arabic/Urdu stopword lists contained markup artifacts** (literal
  `{dir="rtl"}` text baked into each string from a copy-paste origin),
  which meant they could never match real tokens. Cleaned to plain strings.
- **`google.colab` import crashed outside Colab.** Now optional — the
  pipeline saves output locally either way and only triggers the Colab
  download helper when actually running in Colab.
- **No de-duplication before balancing.** One dataset was ~93% duplicate
  rows, badly skewing the "balance to smallest dataset" logic. De-dup now
  happens first, with row counts logged.

## License

Add a license of your choice (MIT is a common default for small projects
like this) — GitHub can generate one for you when you create the repo.
