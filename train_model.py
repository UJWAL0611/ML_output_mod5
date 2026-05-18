# train_model.py
import os
import re
import string
import warnings
import random

import nltk
import pandas as pd
import numpy as np
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ── NLTK downloads ────────────────────────────────────────────────────────────
# Bypass SSL verification on Windows where root certs may be outdated
import ssl as _ssl
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
except AttributeError:
    pass

for resource in ["stopwords", "wordnet", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Map POS treebank tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def preprocess(text: str) -> str:
    """
    Preprocessing pipeline:
    1. Lowercase
    2. Strip HTML tags
    3. Strip URLs
    4. Remove digits
    5. Remove punctuation
    6. Remove English NLTK stopwords
    7. WordNet lemmatize each token (POS-aware)
    8. Drop tokens shorter than 3 chars
    """
    # 1. Lowercase
    text = text.lower()
    # 2. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # 3. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    # 4. Remove digits
    text = re.sub(r"\d+", " ", text)
    # 5. Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # Tokenize
    tokens = word_tokenize(text)
    # POS tag for lemmatization
    tagged = pos_tag(tokens)
    result = []
    for token, tag in tagged:
        # 6. Remove stopwords
        if token in _stop_words:
            continue
        # 7. Lemmatize
        wn_pos = _get_wordnet_pos(tag)
        lemma = _lemmatizer.lemmatize(token, pos=wn_pos)
        # 8. Drop tokens shorter than 3 chars
        if len(lemma) >= 3:
            result.append(lemma)
    return " ".join(result)


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_kaggle_dataset() -> pd.DataFrame:
    """Attempt to download the Kaggle dataset using kaggle.json credentials only."""
    # Only attempt if kaggle.json credentials exist — avoids interactive OAuth prompt
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "~/.kaggle/kaggle.json not found. Skipping Kaggle download."
        )

    import kaggle
    kaggle.api.authenticate()
    os.makedirs("data", exist_ok=True)
    kaggle.api.dataset_download_files(
        "mexwell/fake-reviews-dataset",
        path="data/",
        unzip=True,
        quiet=False,
    )
    # Find the downloaded CSV
    for fname in os.listdir("data"):
        if fname.endswith(".csv"):
            fpath = os.path.join("data", fname)
            df = pd.read_csv(fpath)
            # Normalise column names
            if "text_" in df.columns and "label_" in df.columns:
                df = df.rename(columns={"text_": "review_text", "label_": "label"})
                # CG = Fake (0), OR = Genuine (1)
                df["label"] = df["label"].map({"CG": 0, "OR": 1})
                df = df.dropna(subset=["review_text", "label"])
                df["label"] = df["label"].astype(int)
                return df[["review_text", "label"]]
    raise ValueError("No suitable CSV found after Kaggle download.")


def load_local_dataset() -> pd.DataFrame:
    """Load data/reviews.csv with columns review_text, label."""
    path = os.path.join("data", "reviews.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    if "review_text" not in df.columns or "label" not in df.columns:
        raise ValueError("data/reviews.csv must have columns: review_text, label")
    df = df.dropna(subset=["review_text", "label"])
    df["label"] = df["label"].astype(int)
    return df[["review_text", "label"]]


def generate_synthetic_dataset() -> pd.DataFrame:
    """Generate 300 fake + 300 genuine synthetic reviews."""
    random.seed(42)

    fake_templates = [
        "This product is absolutely the best I have ever used in my entire life!!!",
        "AMAZING AMAZING AMAZING product, changed my life completely!!!",
        "Best purchase ever!!! Totally worth every single penny!!!",
        "I cannot believe how perfect this is, best ever without a doubt!!!",
        "WOW WOW WOW this is incredible, absolutely perfect product!!!",
        "This is the most amazing thing I have ever bought, 100% recommend!!!",
        "BEST PRODUCT EVER!!! Everyone should buy this immediately!!!",
        "Absolutely perfect, changed my life, best ever, highly recommend!!!",
        "I bought this and my life is completely transformed, amazing!!!",
        "This product is a miracle, absolutely the best on the market!!!",
        "Five stars!!! Best ever!!! Buy now!!! You will not regret it!!!",
        "INCREDIBLE product!!! BEST EVER!!! TOTALLY PERFECT!!!",
        "This is hands down the greatest product ever created!!!",
        "Unbelievable quality!!! Best purchase of my life!!!",
        "I have never been so happy with a purchase, absolutely perfect!!!",
    ]

    genuine_templates = [
        "I've been using this product for about three months now. It works as described, though the build quality could be slightly better for the price point.",
        "Decent product overall. Shipping was fast and packaging was secure. The item matches the description fairly well.",
        "I ordered this for my home office setup. It took a few days to arrive and setup was straightforward. Works fine so far.",
        "The product is okay. Not the best I've used but it gets the job done. Customer service was responsive when I had a question.",
        "Bought this as a gift. The recipient seemed happy with it. Quality seems reasonable for the price.",
        "I've had this for two weeks. It does what it says on the box. Nothing extraordinary but no complaints either.",
        "Good value for money. I compared several options before choosing this one. The size is accurate and it fits well.",
        "Works as expected. I had a minor issue initially but it resolved itself. Would consider buying again.",
        "Solid product. I use it daily and it holds up well. The instructions could be clearer but I figured it out.",
        "Reasonable quality at this price range. I've seen better but also much worse. Delivery was on time.",
        "I purchased this after reading several reviews. It meets my needs adequately. The color is slightly different from the photo.",
        "Does the job. Not flashy but functional. I've recommended it to a colleague who was looking for something similar.",
        "Good enough for everyday use. I've had no issues in the first month. Time will tell if it lasts.",
        "The product arrived well packaged. Setup took about 20 minutes. Performance is consistent with what was advertised.",
        "I'm satisfied with this purchase. It's not perfect but it addresses my needs at a fair price.",
    ]

    augment_words_fake = [
        "incredible", "unbelievable", "fantastic", "outstanding", "superb",
        "magnificent", "extraordinary", "phenomenal", "spectacular", "wonderful",
    ]
    augment_words_genuine = [
        "adequate", "reasonable", "acceptable", "satisfactory", "functional",
        "decent", "reliable", "consistent", "practical", "straightforward",
    ]

    rows = []

    for _ in range(300):
        base = random.choice(fake_templates)
        extra = random.choice(augment_words_fake)
        review = base + f" Truly {extra} in every way!"
        rows.append({"review_text": review, "label": 0})

    for _ in range(300):
        base = random.choice(genuine_templates)
        extra = random.choice(augment_words_genuine)
        review = base + f" Overall a {extra} choice."
        rows.append({"review_text": review, "label": 1})

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    os.makedirs("data", exist_ok=True)
    df.to_csv(os.path.join("data", "reviews.csv"), index=False)
    print("[INFO] Synthetic dataset generated and saved to data/reviews.csv")
    return df


def get_dataset() -> pd.DataFrame:
    """Try Kaggle -> local CSV -> synthetic, in that order."""
    # 1. Try Kaggle
    try:
        print("[INFO] Attempting Kaggle download...")
        df = load_kaggle_dataset()
        print(f"[INFO] Kaggle dataset loaded: {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")

    # 2. Try local CSV
    try:
        df = load_local_dataset()
        print(f"[INFO] Local dataset loaded: {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[WARN] Local CSV load failed: {e}")

    # 3. Generate synthetic
    print("[INFO] Generating synthetic dataset...")
    df = generate_synthetic_dataset()
    return df


# ── Main training routine ─────────────────────────────────────────────────────

def main():
    warnings.filterwarnings("ignore")
    os.makedirs("model_artifacts", exist_ok=True)

    # Load data
    df = get_dataset()
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}")

    # Preprocess
    print("[INFO] Preprocessing text (this may take a while)...")
    df["processed"] = df["review_text"].astype(str).apply(preprocess)

    X = df["processed"].values
    y = df["label"].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[INFO] Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Build pipeline
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
    )
    classifier = LogisticRegression(
        C=5.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    pipeline = Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])

    # Train
    print("[INFO] Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULT] Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"[RESULT] 5-Fold CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"         Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Classification report
    print("\n[RESULT] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Genuine"]))

    # Save artefacts separately
    fitted_vectorizer = pipeline.named_steps["tfidf"]
    fitted_classifier = pipeline.named_steps["clf"]

    joblib.dump(fitted_vectorizer, os.path.join("model_artifacts", "tfidf_vectorizer.pkl"), compress=3)
    joblib.dump(fitted_classifier, os.path.join("model_artifacts", "classifier.pkl"), compress=3)
    print("[INFO] Saved model_artifacts/tfidf_vectorizer.pkl")
    print("[INFO] Saved model_artifacts/classifier.pkl")
    print("[INFO] Training complete.")


if __name__ == "__main__":
    main()
