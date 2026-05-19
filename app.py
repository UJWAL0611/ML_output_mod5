# app.py
import os
import io
import re
import json
import string
import logging
import random
import threading
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from flask import Flask, request, jsonify, render_template, send_file
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup

# ── Optional: Playwright (JS-rendering fallback) ───────────────────────────────
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# ── Optional: Google Gemini AI (new SDK) ───────────────────────────────────────
try:
    from google import genai as _genai_module
    GEMINI_AVAILABLE = True
except ImportError:
    _genai_module = None
    GEMINI_AVAILABLE = False

# In-memory Gemini API key (set via /set-api-key endpoint)
_gemini_key_lock = threading.Lock()
_gemini_api_key: str = ""

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── NLTK ──────────────────────────────────────────────────────────────────────
import ssl as _ssl
try:
    _ssl._create_default_https_context = _ssl._create_unverified_context
except AttributeError:
    pass

for _r in ["stopwords", "wordnet", "averaged_perceptron_tagger",
           "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"]:
    nltk.download(_r, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words  = set(stopwords.words("english"))


def _get_wordnet_pos(tag: str) -> str:
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN


def preprocess(text: str) -> str:
    """
    Preprocessing pipeline:
    1. Lowercase  2. Strip HTML  3. Strip URLs  4. Remove digits
    5. Remove punctuation  6. Remove stopwords  7. Lemmatize  8. Drop <3-char tokens
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
        lemma = _lemmatizer.lemmatize(token, pos=_get_wordnet_pos(tag))
        if len(lemma) >= 3:
            result.append(lemma)
    return " ".join(result)


# ── Model loading ─────────────────────────────────────────────────────────────
vectorizer = None
classifier = None
try:
    vectorizer = joblib.load(os.path.join("model_artifacts", "tfidf_vectorizer.pkl"))
    classifier = joblib.load(os.path.join("model_artifacts", "classifier.pkl"))
    logger.info("Model artefacts loaded successfully.")
except Exception as e:
    logger.warning("Could not load model artefacts: %s. Run train_model.py first.", e)


# ══════════════════════════════════════════════════════════════════════════════
# INTELLIGENCE MODULE — Multi-Layer Verification
# ══════════════════════════════════════════════════════════════════════════════

def _compute_linguistic_score(text: str) -> dict:
    """Linguistic pattern analysis — returns score 0-100 (higher = more suspicious)."""
    flags = []
    score = 0
    
    # 1. Exclamation overuse
    exclamations = text.count("!")
    if exclamations >= 5:
        score += 20
        flags.append(f"Excessive exclamation marks ({exclamations})")
    elif exclamations >= 3:
        score += 10
        flags.append(f"High exclamation usage ({exclamations})")
    
    # 2. ALL-CAPS words
    caps_words = len(re.findall(r'\b[A-Z]{3,}\b', text))
    if caps_words >= 5:
        score += 20
        flags.append(f"Excessive ALL-CAPS ({caps_words} words)")
    elif caps_words >= 3:
        score += 10
        flags.append(f"Multiple ALL-CAPS ({caps_words} words)")
    
    # 3. Superlative overload
    superlatives = [
        "best ever", "changed my life", "absolutely perfect", "best purchase",
        "totally worth", "life changing", "never been happier", "most amazing",
        "hands down", "cannot believe", "absolutely incredible", "worth every penny",
        "completely transformed", "best on the market", "totally perfect",
        "game changer", "must buy", "highly recommend", "five stars", "10/10"
    ]
    lower = text.lower()
    matched = [s for s in superlatives if s in lower]
    if len(matched) >= 4:
        score += 25
        flags.append(f"Superlative overload ({len(matched)} phrases)")
    elif len(matched) >= 2:
        score += 12
        flags.append(f"Multiple superlatives ({len(matched)} phrases)")
    
    # 4. Length analysis
    words = text.strip().split()
    word_count = len(words)
    if word_count < 10:
        score += 15
        flags.append(f"Suspiciously short ({word_count} words)")
    elif word_count < 15:
        score += 8
        flags.append(f"Very brief ({word_count} words)")
    
    # 5. Repetition detection
    unique_words = len(set(w.lower() for w in words if len(w) > 3))
    if word_count > 20 and unique_words / word_count < 0.5:
        score += 15
        flags.append("High word repetition")
    
    # 6. Generic template phrases
    templates = [
        "i received this product", "i was sent this", "i got this for free",
        "in exchange for", "honest review", "unbiased review", "my honest opinion",
        "as advertised", "exactly as described", "fast shipping", "arrived quickly"
    ]
    template_matches = [t for t in templates if t in lower]
    if len(template_matches) >= 2:
        score += 18
        flags.append(f"Template language detected ({len(template_matches)} phrases)")
    
    # 7. Lack of specificity (no numbers, no specific details)
    has_numbers = bool(re.search(r'\d', text))
    has_measurements = bool(re.search(r'\d+\s*(cm|mm|inch|kg|lb|oz|ml|liter|hour|day|week|month)', text, re.I))
    if word_count > 30 and not has_numbers and not has_measurements:
        score += 12
        flags.append("Lacks specific details or measurements")
    
    # 8. Emotional intensity (multiple emotional words)
    emotional = ["love", "hate", "amazing", "terrible", "awful", "fantastic", "horrible", "wonderful"]
    emotion_count = sum(1 for e in emotional if e in lower)
    if emotion_count >= 4:
        score += 10
        flags.append(f"High emotional intensity ({emotion_count} emotional words)")
    
    return {
        "score": min(score, 100),
        "flags": flags,
        "word_count": word_count,
        "unique_ratio": round(unique_words / word_count, 2) if word_count > 0 else 0
    }


def _compute_structural_score(text: str) -> dict:
    """Structural analysis — grammar, punctuation, formatting."""
    flags = []
    score = 0
    
    # 1. Sentence structure
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_count = len(sentences)
    
    if sent_count == 1 and len(text.split()) > 30:
        score += 15
        flags.append("Single run-on sentence")
    
    # 2. Punctuation anomalies
    if text.count("...") >= 3:
        score += 10
        flags.append("Excessive ellipsis usage")
    
    if text.count("??") >= 1 or text.count("!!") >= 2:
        score += 12
        flags.append("Repeated punctuation marks")
    
    # 3. Lack of punctuation in long text
    words = text.split()
    if len(words) > 40 and sent_count <= 2:
        score += 15
        flags.append("Long text with minimal punctuation")
    
    # 4. All lowercase or all uppercase
    if text.islower() and len(words) > 15:
        score += 8
        flags.append("Entirely lowercase text")
    elif text.isupper() and len(words) > 10:
        score += 20
        flags.append("Entirely uppercase text")
    
    # 5. Emoji overuse
    emoji_count = len(re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]', text))
    if emoji_count >= 5:
        score += 15
        flags.append(f"Excessive emojis ({emoji_count})")
    elif emoji_count >= 3:
        score += 8
        flags.append(f"Multiple emojis ({emoji_count})")
    
    return {
        "score": min(score, 100),
        "flags": flags,
        "sentence_count": sent_count
    }


def _cross_verify_review(text: str, ml_prediction: dict) -> dict:
    """
    Multi-layer verification combining ML model + heuristics.
    Returns enhanced prediction with trust score and verification breakdown.
    """
    # Get ML prediction confidence
    ml_confidence = ml_prediction["confidence"]
    ml_is_fake = ml_prediction["prediction"] == "Fake"
    
    # Run heuristic analyses
    linguistic = _compute_linguistic_score(text)
    structural = _compute_structural_score(text)
    
    # Compute aggregate suspicion score (0-100, higher = more suspicious)
    heuristic_suspicion = (linguistic["score"] * 0.6 + structural["score"] * 0.4)
    
    # Combine ML and heuristics for final trust score
    # If ML says Fake and heuristics agree (high suspicion) → very low trust
    # If ML says Genuine and heuristics agree (low suspicion) → very high trust
    # If they disagree → moderate trust with warning
    
    if ml_is_fake:
        # ML detected fake — check if heuristics agree
        if heuristic_suspicion >= 50:
            # Strong agreement
            trust_score = max(5, 100 - ml_confidence * 100 - heuristic_suspicion * 0.3)
            verification_status = "high_confidence_fake"
        elif heuristic_suspicion >= 25:
            # Moderate agreement
            trust_score = max(15, 100 - ml_confidence * 100 - heuristic_suspicion * 0.2)
            verification_status = "likely_fake"
        else:
            # Disagreement — ML says fake but heuristics look clean
            trust_score = 40 + (100 - ml_confidence * 100) * 0.3
            verification_status = "uncertain_fake"
    else:
        # ML detected genuine — check if heuristics agree
        if heuristic_suspicion >= 50:
            # Strong disagreement — ML says genuine but heuristics suspicious
            trust_score = 35 - heuristic_suspicion * 0.2
            verification_status = "uncertain_genuine"
        elif heuristic_suspicion >= 25:
            # Moderate disagreement
            trust_score = 55 + ml_confidence * 100 * 0.2 - heuristic_suspicion * 0.3
            verification_status = "likely_genuine"
        else:
            # Strong agreement
            trust_score = 70 + ml_confidence * 100 * 0.3
            verification_status = "high_confidence_genuine"
    
    trust_score = max(0, min(100, trust_score))
    
    # Collect all flags
    all_flags = linguistic["flags"] + structural["flags"]
    
    # Determine final verdict with confidence adjustment
    if verification_status in ["high_confidence_fake", "likely_fake"]:
        final_prediction = "Fake"
        confidence_adjustment = "High confidence — ML and heuristics agree"
    elif verification_status in ["high_confidence_genuine", "likely_genuine"]:
        final_prediction = "Genuine"
        confidence_adjustment = "High confidence — ML and heuristics agree"
    else:
        # Uncertain cases — use ML prediction but flag uncertainty
        final_prediction = ml_prediction["prediction"]
        confidence_adjustment = "Moderate confidence — mixed signals detected"
    
    return {
        "prediction": final_prediction,
        "trust_score": round(trust_score, 1),
        "verification_status": verification_status,
        "confidence_note": confidence_adjustment,
        "ml_confidence": round(ml_confidence * 100, 1),
        "heuristic_suspicion": round(heuristic_suspicion, 1),
        "linguistic_score": linguistic["score"],
        "structural_score": structural["score"],
        "flags": all_flags,
        "word_count": linguistic["word_count"],
        "unique_ratio": linguistic["unique_ratio"],
        "sentence_count": structural["sentence_count"],
        # Keep original ML probabilities
        "fake_prob": ml_prediction["fake_prob"],
        "genuine_prob": ml_prediction["genuine_prob"],
    }


def _predict_single(text: str) -> dict:
    """Enhanced prediction with multi-layer verification."""
    processed = preprocess(text)
    X = vectorizer.transform([processed])
    proba = classifier.predict_proba(X)[0]
    pred_class = int(classifier.predict(X)[0])
    fake_prob    = float(proba[0])
    genuine_prob = float(proba[1])
    
    ml_prediction = {
        "prediction":    "Genuine" if pred_class == 1 else "Fake",
        "confidence":    round(float(max(proba)), 4),
        "confidence_pct": round(float(max(proba)) * 100, 2),
        "fake_prob":     round(fake_prob * 100, 2),
        "genuine_prob":  round(genuine_prob * 100, 2),
    }
    
    # Run cross-verification
    verified = _cross_verify_review(text, ml_prediction)
    
    # Merge results
    return {
        **ml_prediction,
        "trust_score": verified["trust_score"],
        "verification_status": verified["verification_status"],
        "confidence_note": verified["confidence_note"],
        "heuristic_suspicion": verified["heuristic_suspicion"],
        "flags": verified["flags"],
        "analysis": {
            "word_count": verified["word_count"],
            "unique_ratio": verified["unique_ratio"],
            "sentence_count": verified["sentence_count"],
            "linguistic_score": verified["linguistic_score"],
            "structural_score": verified["structural_score"],
        }
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

MAX_REVIEW_CHARS = 10_000
MAX_URL_LENGTH   = 4096
SCRAPE_TIMEOUT   = 20
MAX_REVIEWS      = 50

SCRAPE_HEADERS_POOL = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Sec-Ch-Ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Cache-Control": "max-age=0",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                      "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept-Language": "en-GB,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    },
    {
        "User-Agent": "Mozilla/5.0 (Linux; Android 13; SM-S908B) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
        "Accept-Language": "en-IN,en;q=0.9,hi;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Ch-Ua-Mobile": "?1",
        "Sec-Ch-Ua-Platform": '"Android"',
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# SCRAPING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _fetch(url: str, extra_headers: dict = None):
    """Fetch URL → (BeautifulSoup, raw_text) or (None, None).
    Tries to parse response even on non-200 status if body has content.
    """
    headers = dict(random.choice(SCRAPE_HEADERS_POOL))
    if extra_headers:
        headers.update(extra_headers)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=2)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    try:
        resp = session.get(url, headers=headers, timeout=SCRAPE_TIMEOUT,
                           verify=False, allow_redirects=True)
        logger.info("Fetched %s — HTTP %s, %d bytes", url[:80], resp.status_code, len(resp.content))
        # Even on non-200 status, try to parse if we got substantial HTML content
        if resp.status_code == 200 or (resp.text and len(resp.text) > 5000):
            return BeautifulSoup(resp.text, "lxml"), resp.text
        else:
            logger.warning("HTTP %s for %s (body too small to parse: %d bytes)",
                           resp.status_code, url[:60], len(resp.content))
            return None, None
    except requests.exceptions.ConnectionError as e:
        logger.warning("Connection error %s: %s", url[:60], e)
    except requests.exceptions.Timeout:
        logger.warning("Timeout %s", url[:60])
    except Exception as e:
        logger.warning("Fetch error %s: %s", url[:60], e)
    return None, None


def _clean(texts: list, min_words: int = 6, max_words: int = 500) -> list:
    """Deduplicate and filter by word count."""
    seen, out = set(), []
    for t in texts:
        t = re.sub(r"\s+", " ", t).strip()
        wc = len(t.split())
        key = t[:120]
        if min_words <= wc <= max_words and key not in seen:
            seen.add(key)
            out.append(t)
    return out[:MAX_REVIEWS]


def _extract_json_reviews(raw: str) -> list:
    """Pull review strings out of JSON blobs embedded in page source."""
    texts = []
    for key in ["reviewText", "reviewBody", "review_text", "body",
                "content", "text", "description", "comment"]:
        for m in re.finditer(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)+)"', raw):
            val = m.group(1).encode().decode("unicode_escape", errors="ignore")
            val = re.sub(r"<[^>]+>", " ", val).strip()
            if len(val.split()) >= 6:
                texts.append(val)
    return texts


def _extract_schema(soup: BeautifulSoup) -> list:
    """schema.org itemprop=reviewBody."""
    texts = []
    for el in soup.find_all(attrs={"itemprop": re.compile(r"reviewBody|description", re.I)}):
        t = el.get_text(separator=" ", strip=True)
        if len(t.split()) >= 6:
            texts.append(t)
    return texts


def _extract_keywords(soup: BeautifulSoup) -> list:
    """Keyword-based class/id/data-* hunt."""
    kw = re.compile(
        r"review|comment|feedback|rating|testimonial|opinion|"
        r"customer.?say|buyer|verified|purchase", re.I
    )
    texts = []
    for el in soup.find_all(["div","li","article","section","p","span"], class_=kw):
        texts.append(el.get_text(separator=" ", strip=True))
    for el in soup.find_all(id=kw):
        texts.append(el.get_text(separator=" ", strip=True))
    for el in soup.find_all(attrs={"data-hook": re.compile(r"review", re.I)}):
        texts.append(el.get_text(separator=" ", strip=True))
    for el in soup.find_all(attrs={"data-review-id": True}):
        texts.append(el.get_text(separator=" ", strip=True))
    return texts


def _extract_paragraphs(soup: BeautifulSoup) -> list:
    """Last-resort paragraph extraction."""
    return [el.get_text(separator=" ", strip=True)
            for el in soup.find_all(["p", "li"])
            if 8 <= len(el.get_text(strip=True).split()) <= 300]


def _smart_scrape(url: str, extra_headers: dict = None) -> list:
    """Multi-strategy scraper: JSON → schema.org → keywords → paragraphs."""
    soup, raw = _fetch(url, extra_headers)
    if soup is None:
        return []
    for tag in soup(["nav","header","footer","script","style","aside","noscript","iframe","form"]):
        tag.decompose()
    texts = []
    if raw:
        texts += _extract_json_reviews(raw)
    texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    if not texts:
        texts += _extract_paragraphs(soup)
    return _clean(texts)


# ── Platform scrapers (defined BEFORE routing table) ─────────────────────────

def _scrape_amazon(url: str) -> list:
    base = re.match(r"(https?://[^/]+)", url)
    domain = base.group(1) if base else "https://www.amazon.com"

    asin_match = re.search(r"/(?:dp|gp/product|product-reviews)/([A-Z0-9]{10})", url)
    if not asin_match:
        # Try extracting ASIN from anywhere in the URL path
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url)

    texts = []

    if asin_match:
        asin = asin_match.group(1)
        # Try multiple review URL formats
        candidate_urls = [
            f"{domain}/product-reviews/{asin}?reviewerType=all_reviews&pageSize=20",
            f"{domain}/product-reviews/{asin}",
            f"https://www.amazon.com/product-reviews/{asin}?reviewerType=all_reviews&pageSize=20",
        ]
        for fetch_url in candidate_urls:
            soup, raw = _fetch(fetch_url)
            if soup is None:
                continue
            # Amazon-specific: data-hook="review-body"
            found = [el.get_text(separator=" ", strip=True)
                     for el in soup.find_all(attrs={"data-hook": "review-body"})]
            if not found:
                found = [el.get_text(separator=" ", strip=True)
                         for el in soup.find_all(["div", "span"],
                             class_=re.compile(r"review-text|reviewText|review-body", re.I))]
            if not found and raw:
                found = _extract_json_reviews(raw)
            if not found:
                found = _extract_schema(soup)
            if not found:
                found = _extract_keywords(soup)
            if found:
                texts = found
                break
    else:
        # No ASIN found — scrape the URL as-is
        soup, raw = _fetch(url)
        if soup:
            texts = [el.get_text(separator=" ", strip=True)
                     for el in soup.find_all(attrs={"data-hook": "review-body"})]
            if not texts and raw:
                texts = _extract_json_reviews(raw)
            if not texts:
                texts = _extract_schema(soup)
            if not texts:
                texts = _extract_keywords(soup)

    return _clean(texts)


def _scrape_flipkart(url: str) -> list:
    reviews_url = url
    pid = re.search(r"pid=([A-Z0-9]+)", url)
    if pid and "/p/" in url and "reviews" not in url.lower():
        base = re.match(r"(https?://[^/]+/[^?]+)", url)
        if base:
            reviews_url = base.group(1) + "?pid=" + pid.group(1) + "#reviews"
    soup, raw = _fetch(reviews_url)
    if soup is None:
        return []
    texts = []
    if raw:
        texts += _extract_json_reviews(raw)
    texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    if not texts:
        texts += _extract_paragraphs(soup)
    return _clean(texts)


def _scrape_ebay(url: str) -> list:
    soup, raw = _fetch(url)
    if soup is None:
        return []
    texts = [el.get_text(separator=" ", strip=True)
             for el in soup.find_all(["p","div","span"],
                 class_=re.compile(r"review-item-content|rvw-review-text|review-content|ebay-review", re.I))]
    if not texts and raw:
        texts += _extract_json_reviews(raw)
    if not texts:
        texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    return _clean(texts)


def _scrape_walmart(url: str) -> list:
    soup, raw = _fetch(url)
    if soup is None:
        return []
    texts = [el.get_text(separator=" ", strip=True)
             for el in soup.find_all(["div","span","p"],
                 class_=re.compile(r"review-text|ReviewBody|review-body|review-content", re.I))]
    if not texts and raw:
        texts += _extract_json_reviews(raw)
    if not texts:
        texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    return _clean(texts)


def _scrape_trustpilot(url: str) -> list:
    soup, raw = _fetch(url)
    if soup is None:
        return []
    texts = [el.get_text(separator=" ", strip=True)
             for el in soup.find_all(attrs={"data-service-review-text-typography": True})]
    if not texts:
        texts = [el.get_text(separator=" ", strip=True)
                 for el in soup.find_all(["p"],
                     class_=re.compile(r"typography_body|review-content__text", re.I))]
    if not texts and raw:
        texts += _extract_json_reviews(raw)
    if not texts:
        texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    return _clean(texts)


def _scrape_tripadvisor(url: str) -> list:
    soup, raw = _fetch(url)
    if soup is None:
        return []
    texts = [el.get_text(separator=" ", strip=True)
             for el in soup.find_all(["q","p","div"],
                 class_=re.compile(r"QewHA|partial_entry|review-container|biGQs", re.I))]
    if not texts and raw:
        texts += _extract_json_reviews(raw)
    if not texts:
        texts += _extract_schema(soup)
    if not texts:
        texts += _extract_keywords(soup)
    return _clean(texts)


# All remaining platforms use the universal smart scraper
def _scrape_meesho(url):      return _smart_scrape(url)
def _scrape_myntra(url):      return _smart_scrape(url)
def _scrape_snapdeal(url):    return _smart_scrape(url)
def _scrape_bestbuy(url):     return _smart_scrape(url)
def _scrape_target(url):      return _smart_scrape(url)
def _scrape_etsy(url):        return _smart_scrape(url)
def _scrape_aliexpress(url):  return _smart_scrape(url, {"Referer": "https://www.aliexpress.com/"})
def _scrape_alibaba(url):     return _smart_scrape(url)
def _scrape_jd(url):          return _smart_scrape(url, {"Referer": "https://www.jd.com/"})
def _scrape_lazada(url):      return _smart_scrape(url)
def _scrape_shopee(url):      return _smart_scrape(url)
def _scrape_noon(url):        return _smart_scrape(url)
def _scrape_jumia(url):       return _smart_scrape(url)
def _scrape_temu(url):        return _smart_scrape(url)
def _scrape_shein(url):       return _smart_scrape(url)
def _scrape_newegg(url):      return _smart_scrape(url)
def _scrape_costco(url):      return _smart_scrape(url)
def _scrape_yelp(url):        return _smart_scrape(url)
def _scrape_google(url):      return _smart_scrape(url)
def _scrape_generic(url):     return _smart_scrape(url)


def _playwright_scrape(url: str) -> list:
    """JS-rendering fallback using Playwright headless Chromium.
    Auto-scrolls to trigger lazy-loaded review sections.
    Works for Flipkart, Meesho, Myntra, Snapdeal, and any JS-heavy site.
    """
    if not PLAYWRIGHT_AVAILABLE:
        logger.warning("Playwright not installed — JS fallback unavailable.")
        return []
    logger.info("Playwright JS-render: %s", url[:80])
    texts = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0.0.0 Safari/537.36",
                viewport={"width": 1280, "height": 900},
                locale="en-US",
            )
            page = ctx.new_page()
            page.goto(url, timeout=30000, wait_until="domcontentloaded")
            # Scroll down in steps to trigger lazy-loading
            for _ in range(5):
                page.evaluate("window.scrollBy(0, window.innerHeight)")
                page.wait_for_timeout(1000)
            html = page.content()
            browser.close()

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["nav", "header", "footer", "script", "style", "aside", "noscript", "iframe"]):
            tag.decompose()

        texts += _extract_json_reviews(html)
        texts += _extract_schema(soup)
        if not texts:
            texts += _extract_keywords(soup)
        if not texts:
            texts += _extract_paragraphs(soup)
    except Exception as e:
        logger.warning("Playwright scrape failed: %s", e)
    return _clean(texts)


# ── Platform routing table (after all scrapers are defined) ───────────────────
PLATFORM_SCRAPERS = [
    ("amazon.in",      _scrape_amazon),
    ("flipkart.",      _scrape_flipkart),
    ("meesho.",        _scrape_meesho),
    ("myntra.",        _scrape_myntra),
    ("snapdeal.",      _scrape_snapdeal),
    ("amazon.",        _scrape_amazon),
    ("walmart.",       _scrape_walmart),
    ("bestbuy.",       _scrape_bestbuy),
    ("target.",        _scrape_target),
    ("costco.",        _scrape_costco),
    ("newegg.",        _scrape_newegg),
    ("ebay.",          _scrape_ebay),
    ("etsy.",          _scrape_etsy),
    ("aliexpress.",    _scrape_aliexpress),
    ("alibaba.",       _scrape_alibaba),
    ("temu.",          _scrape_temu),
    ("shein.",         _scrape_shein),
    ("jd.com",         _scrape_jd),
    ("jd.id",          _scrape_jd),
    ("lazada.",        _scrape_lazada),
    ("shopee.",        _scrape_shopee),
    ("noon.",          _scrape_noon),
    ("jumia.",         _scrape_jumia),
    ("trustpilot.",    _scrape_trustpilot),
    ("yelp.",          _scrape_yelp),
    ("tripadvisor.",   _scrape_tripadvisor),
    ("google.",        _scrape_google),
]


def scrape_reviews(url: str) -> list:
    lower = url.lower()
    specialist = next((fn for frag, fn in PLATFORM_SCRAPERS if frag in lower), None)
    reviews = []
    if specialist:
        logger.info("Specialist: %s → %s", specialist.__name__, url[:60])
        reviews = specialist(url)
    if not reviews:
        logger.info("Generic fallback → %s", url[:60])
        reviews = _scrape_generic(url)
    # If static scraping returned nothing, try JS-rendering with Playwright
    if not reviews and PLAYWRIGHT_AVAILABLE:
        logger.info("Trying Playwright JS-render fallback → %s", url[:60])
        reviews = _playwright_scrape(url)
    logger.info("Total scraped: %d reviews", len(reviews))
    return reviews


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI AI INTELLIGENCE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _gemini_verify_reviews(texts: list) -> list:
    """
    Send a batch of reviews to Gemini for AI-powered authenticity verification.
    Returns a list of dicts: [{authentic: bool, confidence: 0-100, reason: str}]
    Falls back to empty list if Gemini is unavailable or key not set.
    """
    global _gemini_api_key
    with _gemini_key_lock:
        key = _gemini_api_key

    if not GEMINI_AVAILABLE or not key or _genai_module is None:
        return []

    try:
        client = _genai_module.Client(api_key=key)

        # Build compact numbered list for the prompt
        numbered = "\n".join(
            f"[{i+1}] {t[:400]}" for i, t in enumerate(texts[:15])
        )
        prompt = (
            "You are an expert at detecting fake product reviews. "
            "Analyse each review below and respond ONLY with a valid JSON array "
            "(no markdown, no explanation) where each element has:\n"
            "  \"index\": (1-based integer)\n"
            "  \"authentic\": true or false\n"
            "  \"confidence\": integer 0-100 (how confident you are)\n"
            "  \"reason\": very short reason (max 12 words)\n\n"
            "Reviews:\n" + numbered
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        raw = resp.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.S).strip()
        parsed = json.loads(raw)

        # Normalise to list indexed by 0
        result = [None] * len(texts[:15])
        for item in parsed:
            idx = int(item.get("index", 0)) - 1
            if 0 <= idx < len(result):
                result[idx] = {
                    "authentic": bool(item.get("authentic", True)),
                    "confidence": int(item.get("confidence", 50)),
                    "reason": str(item.get("reason", ""))[:120],
                }
        # Fill any missing slots with neutral defaults
        for i in range(len(result)):
            if result[i] is None:
                result[i] = {"authentic": True, "confidence": 50, "reason": "No AI verdict"}
        return result
    except Exception as e:
        logger.warning("Gemini verification failed: %s", e)
        return []





# ══════════════════════════════════════════════════════════════════════════════
# FLASK ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """Return a minimal 1x1 transparent ICO to suppress browser 404."""
    # Minimal valid 1x1 transparent ICO file (62 bytes)
    ico = (
        b'\x00\x00\x01\x00\x01\x00\x01\x01\x00\x00\x01\x00\x18\x00'
        b'\x30\x00\x00\x00\x16\x00\x00\x00\x28\x00\x00\x00\x01\x00'
        b'\x00\x00\x02\x00\x00\x00\x01\x00\x18\x00\x00\x00\x00\x00'
        b'\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    )
    return send_file(io.BytesIO(ico), mimetype="image/x-icon")


@app.route("/health", methods=["GET"])
def health():
    if vectorizer is not None and classifier is not None:
        return jsonify({"status": "ready"}), 200
    return jsonify({"status": "model_not_loaded"}), 200


@app.route("/set-api-key", methods=["POST"])
def set_api_key():
    """Store the Gemini API key in memory for this session."""
    global _gemini_api_key
    body = request.get_json(silent=True) or {}
    key = (body.get("api_key") or "").strip()
    with _gemini_key_lock:
        _gemini_api_key = key
    if key:
        logger.info("Gemini API key set (%d chars).", len(key))
        return jsonify({"status": "ok", "gemini_active": True}), 200
    else:
        logger.info("Gemini API key cleared.")
        return jsonify({"status": "ok", "gemini_active": False}), 200


@app.route("/gemini-status", methods=["GET"])
def gemini_status():
    """Return whether Gemini AI is available and a key is set."""
    with _gemini_key_lock:
        has_key = bool(_gemini_api_key)
    return jsonify({
        "gemini_library": GEMINI_AVAILABLE,
        "playwright_library": PLAYWRIGHT_AVAILABLE,
        "gemini_active": GEMINI_AVAILABLE and has_key,
    }), 200




@app.route("/predict", methods=["POST"])
def predict():
    if vectorizer is None or classifier is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    body = (request.get_json(silent=True) or {}) if request.is_json else {}
    review_text = body.get("review_text", "") or request.form.get("review_text", "")

    if not isinstance(review_text, str) or not review_text.strip():
        return jsonify({"error": "review_text must be a non-empty string."}), 400
    if len(review_text) > MAX_REVIEW_CHARS:
        return jsonify({"error": f"review_text exceeds {MAX_REVIEW_CHARS} characters."}), 400

    try:
        result = _predict_single(review_text)

        # Add Gemini AI cross-verification for single reviews too
        with _gemini_key_lock:
            gemini_active = bool(_gemini_api_key) and GEMINI_AVAILABLE

        if gemini_active:
            verdicts = _gemini_verify_reviews([review_text])
            if verdicts and verdicts[0]:
                v = verdicts[0]
                result["ai_verdict"] = v
                # Blend: 60% ML+heuristic trust, 40% Gemini
                ai_trust = v["confidence"] if v["authentic"] else (100 - v["confidence"])
                old_trust = result.get("trust_score", 50)
                result["trust_score"] = round(old_trust * 0.60 + ai_trust * 0.40, 1)
                # Conflict flag
                ml_fake = result["prediction"] == "Fake"
                ai_fake = not v["authentic"]
                if ml_fake != ai_fake and v["confidence"] >= 70:
                    result["verification_status"] = "ai_conflict"
        else:
            result["ai_verdict"] = None

        result["gemini_active"] = gemini_active
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Prediction error.")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500




def _detect_aggregate_patterns(results: list) -> dict:
    """
    Analyze patterns across multiple reviews to detect coordinated manipulation.
    Returns suspicious patterns and aggregate trust indicators.
    """
    if not results:
        return {"patterns": [], "aggregate_trust": 50}
    
    patterns = []
    
    # 1. Similarity clustering — detect copy-paste reviews
    texts = [r.get("text", "") for r in results]
    if len(texts) >= 3:
        # Simple similarity check: count reviews with >70% word overlap
        similar_pairs = 0
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                words_i = set(texts[i].lower().split())
                words_j = set(texts[j].lower().split())
                if len(words_i) > 5 and len(words_j) > 5:
                    overlap = len(words_i & words_j) / min(len(words_i), len(words_j))
                    if overlap > 0.7:
                        similar_pairs += 1
        
        if similar_pairs >= 2:
            patterns.append(f"⚠️ {similar_pairs} pairs of highly similar reviews detected (possible copy-paste)")
    
    # 2. Length uniformity — all reviews suspiciously similar length
    word_counts = [len(r.get("text", "").split()) for r in results]
    if len(word_counts) >= 5:
        avg_len = sum(word_counts) / len(word_counts)
        variance = sum((wc - avg_len) ** 2 for wc in word_counts) / len(word_counts)
        std_dev = variance ** 0.5
        if std_dev < 5 and avg_len > 20:
            patterns.append(f"⚠️ Suspiciously uniform review lengths (avg {int(avg_len)} words, σ={int(std_dev)})")
    
    # 3. Extreme rating distribution
    fake_count = sum(1 for r in results if r.get("prediction") == "Fake")
    genuine_count = len(results) - fake_count
    fake_pct = fake_count / len(results) * 100
    
    if fake_pct >= 80:
        patterns.append(f"🚨 Overwhelming fake reviews ({fake_pct:.0f}%) — likely coordinated campaign")
    elif fake_pct >= 60:
        patterns.append(f"⚠️ High fake review ratio ({fake_pct:.0f}%) — product credibility questionable")
    
    # 4. Trust score clustering — all reviews have similar confidence
    trust_scores = [r.get("trust_score", 50) for r in results if "trust_score" in r]
    if len(trust_scores) >= 5:
        avg_trust = sum(trust_scores) / len(trust_scores)
        trust_variance = sum((ts - avg_trust) ** 2 for ts in trust_scores) / len(trust_scores)
        trust_std = trust_variance ** 0.5
        if trust_std < 8:
            patterns.append(f"⚠️ Uniform trust scores (σ={trust_std:.1f}) — possible bot-generated content")
    
    # 5. Flag concentration — many reviews share same heuristic flags
    all_flags = []
    for r in results:
        all_flags.extend(r.get("flags", []))
    
    if all_flags:
        from collections import Counter
        flag_counts = Counter(all_flags)
        common_flags = [f for f, count in flag_counts.items() if count >= len(results) * 0.4]
        if common_flags:
            patterns.append(f"⚠️ Repeated patterns: {', '.join(common_flags[:2])}")
    
    # Compute aggregate trust score
    if trust_scores:
        aggregate_trust = sum(trust_scores) / len(trust_scores)
    else:
        # Fallback based on genuine percentage
        aggregate_trust = genuine_count / len(results) * 100
    
    # Adjust aggregate trust based on detected patterns
    pattern_penalty = min(len(patterns) * 8, 30)
    aggregate_trust = max(0, aggregate_trust - pattern_penalty)
    
    return {
        "patterns": patterns,
        "aggregate_trust": round(aggregate_trust, 1),
        "similarity_detected": any("similar" in p.lower() for p in patterns),
        "uniformity_detected": any("uniform" in p.lower() for p in patterns),
    }


@app.route("/analyse-url", methods=["POST"])
def analyse_url():
    if vectorizer is None or classifier is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    body = request.get_json(silent=True) or {}
    url  = (body.get("url") or "").strip().strip('"\'')

    if not url:
        return jsonify({"error": "url field is required."}), 400
    if len(url) > MAX_URL_LENGTH:
        return jsonify({"error": f"URL too long (max {MAX_URL_LENGTH} chars)."}), 400
    if not re.match(r"https?://", url, re.I):
        url = "https://" + url

    logger.info("Scanning URL: %s", url[:120])

    try:
        reviews_raw = scrape_reviews(url)
    except Exception as e:
        logger.exception("Scraping failed.")
        return jsonify({"error": f"Could not scrape URL: {str(e)}"}), 502

    playwright_note = PLAYWRIGHT_AVAILABLE

    if not reviews_raw:
        tips = (
            "No reviews could be extracted from that page. Common reasons:\n"
            "• The site requires login or shows a CAPTCHA\n"
            "• The page has no reviews yet\n"
            "• Reviews load via JavaScript (try the reviews/ratings tab URL directly)\n\n"
            "Tips:\n"
            "• Paste the URL of the product's reviews page, not the main product page\n"
            "• For Amazon — use amazon.com/product-reviews/ASIN\n"
            "• For Flipkart/Meesho — paste the product page URL and let JS-rendering try\n"
            "• Trustpilot, Yelp, TripAdvisor work best with their review listing pages\n"
        )
        if not PLAYWRIGHT_AVAILABLE:
            tips += "• Install Playwright (pip install playwright && playwright install chromium) for JS sites\n"
        return jsonify({"error": tips}), 200

    # ── ML classification ──────────────────────────────────────────────────────
    results, genuine_count, fake_count = [], 0, 0
    for text in reviews_raw:
        try:
            pred = _predict_single(text)
            results.append({
                "text":           text[:500],
                "prediction":     pred["prediction"],
                "confidence_pct": pred["confidence_pct"],
                "fake_prob":      pred["fake_prob"],
                "genuine_prob":   pred["genuine_prob"],
                "trust_score":    pred.get("trust_score", 50),
                "flags":          pred.get("flags", []),
                "verification_status": pred.get("verification_status", "unknown"),
                "ai_verdict":     None,
            })
            if pred["prediction"] == "Genuine":
                genuine_count += 1
            else:
                fake_count += 1
        except Exception:
            continue

    total = len(results)
    if total == 0:
        return jsonify({"error": "Reviews found but could not be classified."}), 500

    # ── Gemini AI cross-verification ───────────────────────────────────────────
    with _gemini_key_lock:
        gemini_active = bool(_gemini_api_key) and GEMINI_AVAILABLE

    if gemini_active:
        raw_texts = [r["text"] for r in results]
        gemini_verdicts = _gemini_verify_reviews(raw_texts)
        for i, verdict in enumerate(gemini_verdicts):
            if i < len(results) and verdict:
                results[i]["ai_verdict"] = verdict
                # Blend trust score: 60% ML+heuristic, 40% Gemini
                ai_conf = verdict["confidence"]
                ai_trust = ai_conf if verdict["authentic"] else (100 - ai_conf)
                results[i]["trust_score"] = round(results[i]["trust_score"] * 0.60 + ai_trust * 0.40, 1)
                # Flag AI conflict (ML and Gemini strongly disagree)
                ml_fake = results[i]["prediction"] == "Fake"
                ai_fake = not verdict["authentic"]
                if ml_fake != ai_fake and ai_conf >= 70:
                    results[i]["verification_status"] = "ai_conflict"

    # ── Aggregate pattern detection ────────────────────────────────────────────
    aggregate_analysis = _detect_aggregate_patterns(results)

    return jsonify({
        "url":             url,
        "total":           total,
        "genuine_count":   genuine_count,
        "fake_count":      fake_count,
        "genuine_pct":     round(genuine_count / total * 100, 1),
        "fake_pct":        round(fake_count    / total * 100, 1),
        "reviews":         results,
        "aggregate_trust": aggregate_analysis["aggregate_trust"],
        "suspicious_patterns": aggregate_analysis["patterns"],
        "pattern_flags": {
            "similarity_detected": aggregate_analysis["similarity_detected"],
            "uniformity_detected": aggregate_analysis["uniformity_detected"],
        },
        "gemini_active":     gemini_active,
        "playwright_active": PLAYWRIGHT_AVAILABLE,
    }), 200


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

