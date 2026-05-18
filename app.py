# app.py
import os
import re
import string
import logging
import time
import random

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── NLTK downloads ────────────────────────────────────────────────────────────
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
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    result = []
    for token, tag in tagged:
        if token in _stop_words:
            continue
        wn_pos = _get_wordnet_pos(tag)
        lemma = _lemmatizer.lemmatize(token, pos=wn_pos)
        if len(lemma) >= 3:
            result.append(lemma)
    return " ".join(result)


# ── Model loading ─────────────────────────────────────────────────────────────
VECTORIZER_PATH = os.path.join("model_artifacts", "tfidf_vectorizer.pkl")
CLASSIFIER_PATH = os.path.join("model_artifacts", "classifier.pkl")

vectorizer = None
classifier = None

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    logger.info("Model artefacts loaded successfully.")
except Exception as e:
    logger.warning(
        "Could not load model artefacts: %s. Run train_model.py first.", e
    )

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

MAX_REVIEW_CHARS = 10_000

# ── Browser-like headers for scraping ────────────────────────────────────────
SCRAPE_HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                      "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    },
]


def _predict_single(text: str) -> dict:
    """Run model on one review text. Returns dict with prediction fields."""
    processed = preprocess(text)
    X = vectorizer.transform([processed])
    proba = classifier.predict_proba(X)[0]
    pred_class = int(classifier.predict(X)[0])
    fake_prob = float(proba[0])
    genuine_prob = float(proba[1])
    return {
        "prediction": "Genuine" if pred_class == 1 else "Fake",
        "confidence": round(float(max(proba)), 4),
        "confidence_pct": round(float(max(proba)) * 100, 2),
        "fake_prob": round(fake_prob * 100, 2),
        "genuine_prob": round(genuine_prob * 100, 2),
    }


# ── Scrapers ──────────────────────────────────────────────────────────────────

def _scrape_amazon(url: str) -> list[str]:
    """Scrape review texts from an Amazon product page."""
    reviews = []
    headers = random.choice(SCRAPE_HEADERS_POOL)

    # Normalise to product ASIN reviews page
    asin_match = re.search(r"/dp/([A-Z0-9]{10})", url)
    if asin_match:
        asin = asin_match.group(1)
        # Use the all-reviews page
        base = re.match(r"(https?://[^/]+)", url)
        domain = base.group(1) if base else "https://www.amazon.com"
        fetch_url = f"{domain}/product-reviews/{asin}?reviewerType=all_reviews&pageSize=20"
    else:
        fetch_url = url

    try:
        resp = requests.get(fetch_url, headers=headers, timeout=12, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Amazon review body selectors (multiple fallbacks)
        selectors = [
            {"data-hook": "review-body"},
        ]
        for sel in selectors:
            spans = soup.find_all("span", attrs=sel)
            for span in spans:
                text = span.get_text(separator=" ", strip=True)
                if len(text.split()) >= 5:
                    reviews.append(text)

        # Fallback: class-based
        if not reviews:
            for div in soup.find_all("div", class_=re.compile(r"review-text")):
                text = div.get_text(separator=" ", strip=True)
                if len(text.split()) >= 5:
                    reviews.append(text)

    except Exception as e:
        logger.warning("Amazon scrape error: %s", e)

    return reviews[:50]  # cap at 50


def _scrape_flipkart(url: str) -> list[str]:
    """Scrape review texts from a Flipkart product page."""
    reviews = []
    headers = random.choice(SCRAPE_HEADERS_POOL)

    try:
        resp = requests.get(url, headers=headers, timeout=12, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Flipkart review selectors
        for div in soup.find_all("div", class_=re.compile(r"t-ZTKy|_6K-7Co|row")):
            text = div.get_text(separator=" ", strip=True)
            if 10 <= len(text.split()) <= 300:
                reviews.append(text)

        # Fallback: paragraph tags inside review containers
        if not reviews:
            for p in soup.find_all("p", class_=re.compile(r"_2-N8zT|review")):
                text = p.get_text(separator=" ", strip=True)
                if len(text.split()) >= 5:
                    reviews.append(text)

    except Exception as e:
        logger.warning("Flipkart scrape error: %s", e)

    return reviews[:50]


def _scrape_generic(url: str) -> list[str]:
    """Generic scraper — extracts paragraph-length text blocks that look like reviews."""
    reviews = []
    headers = random.choice(SCRAPE_HEADERS_POOL)

    try:
        resp = requests.get(url, headers=headers, timeout=12, verify=False)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove nav/header/footer noise
        for tag in soup(["nav", "header", "footer", "script", "style", "aside"]):
            tag.decompose()

        # Look for review-like containers by common class keywords
        review_keywords = re.compile(
            r"review|comment|feedback|rating|testimonial|opinion", re.I
        )
        candidates = soup.find_all(
            ["div", "li", "article", "section", "p"],
            class_=review_keywords
        )
        for el in candidates:
            text = el.get_text(separator=" ", strip=True)
            words = text.split()
            if 8 <= len(words) <= 400:
                reviews.append(text)

        # If still nothing, grab all <p> tags of reasonable length
        if not reviews:
            for p in soup.find_all("p"):
                text = p.get_text(separator=" ", strip=True)
                words = text.split()
                if 8 <= len(words) <= 300:
                    reviews.append(text)

    except Exception as e:
        logger.warning("Generic scrape error: %s", e)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for r in reviews:
        key = r[:80]
        if key not in seen:
            seen.add(key)
            unique.append(r)

    return unique[:50]


def scrape_reviews(url: str) -> list[str]:
    """Route URL to the right scraper."""
    lower = url.lower()
    if "amazon." in lower:
        reviews = _scrape_amazon(url)
    elif "flipkart." in lower:
        reviews = _scrape_flipkart(url)
    else:
        reviews = _scrape_generic(url)

    # If specialised scraper returned nothing, fall back to generic
    if not reviews and "amazon." not in lower and "flipkart." not in lower:
        reviews = _scrape_generic(url)
    elif not reviews:
        reviews = _scrape_generic(url)

    return reviews


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    if vectorizer is not None and classifier is not None:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "model_not_loaded"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Analyse a single pasted review."""
    if vectorizer is None or classifier is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    if request.is_json:
        body = request.get_json(silent=True) or {}
        review_text = body.get("review_text", "")
    else:
        review_text = request.form.get("review_text", "")

    if not isinstance(review_text, str) or not review_text.strip():
        return jsonify({"error": "review_text must be a non-empty string."}), 400

    if len(review_text) > MAX_REVIEW_CHARS:
        return jsonify({
            "error": f"review_text exceeds {MAX_REVIEW_CHARS} characters."
        }), 400

    try:
        return jsonify(_predict_single(review_text)), 200
    except Exception as e:
        logger.exception("Prediction error.")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/analyse-url", methods=["POST"])
def analyse_url():
    """
    Scrape reviews from a product URL and return aggregate stats.
    Body: { "url": "https://..." }
    Response: {
        "url": str,
        "total": int,
        "genuine_count": int,
        "fake_count": int,
        "genuine_pct": float,
        "fake_pct": float,
        "reviews": [ { "text": str, "prediction": str, "confidence_pct": float,
                        "fake_prob": float, "genuine_prob": float }, ... ]
    }
    """
    if vectorizer is None or classifier is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    body = request.get_json(silent=True) or {}
    url = (body.get("url") or "").strip()

    if not url:
        return jsonify({"error": "url field is required."}), 400

    if not re.match(r"https?://", url, re.I):
        return jsonify({"error": "url must start with http:// or https://"}), 400

    try:
        reviews_raw = scrape_reviews(url)
    except Exception as e:
        logger.exception("Scraping failed.")
        return jsonify({"error": f"Could not scrape URL: {str(e)}"}), 502

    if not reviews_raw:
        return jsonify({
            "error": (
                "No reviews found on that page. "
                "The site may block scrapers or the URL may not be a product/review page. "
                "Try the direct reviews URL (e.g. amazon.com/product-reviews/ASIN)."
            )
        }), 404

    results = []
    genuine_count = 0
    fake_count = 0

    for text in reviews_raw:
        try:
            pred = _predict_single(text)
            results.append({
                "text": text[:500],   # truncate for response size
                "prediction": pred["prediction"],
                "confidence_pct": pred["confidence_pct"],
                "fake_prob": pred["fake_prob"],
                "genuine_prob": pred["genuine_prob"],
            })
            if pred["prediction"] == "Genuine":
                genuine_count += 1
            else:
                fake_count += 1
        except Exception:
            continue

    total = len(results)
    if total == 0:
        return jsonify({"error": "Reviews were found but could not be classified."}), 500

    genuine_pct = round(genuine_count / total * 100, 1)
    fake_pct = round(fake_count / total * 100, 1)

    return jsonify({
        "url": url,
        "total": total,
        "genuine_count": genuine_count,
        "fake_count": fake_count,
        "genuine_pct": genuine_pct,
        "fake_pct": fake_pct,
        "reviews": results,
    }), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
