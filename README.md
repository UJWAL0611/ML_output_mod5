<![CDATA[# 🛡️ ReviewGuard — Universal Fake Review Detector

> **Detect fake product reviews instantly** using a multi-layered intelligence engine powered by Machine Learning, NLP heuristics, and Google Gemini AI.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.1-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-3.9-154F5B)
![Gemini](https://img.shields.io/badge/Gemini_AI-2.0_Flash-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Train the Model](#train-the-model)
  - [Run the Application](#run-the-application)
- [Usage](#usage)
  - [URL Scan Mode](#url-scan-mode)
  - [Single Review Mode](#single-review-mode)
  - [Gemini AI Setup (Optional)](#gemini-ai-setup-optional)
- [API Endpoints](#api-endpoints)
- [Supported Platforms](#supported-platforms)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

**ReviewGuard** is a full-stack web application that identifies fake and deceptive product reviews across any e-commerce platform. It combines a **Logistic Regression ML classifier** with **linguistic heuristic analysis** and optional **Google Gemini AI cross-verification** to deliver high-confidence authenticity verdicts for individual reviews and bulk URL scans.

The system supports **26+ e-commerce platforms** out of the box — including Amazon, Flipkart, Meesho, Myntra, eBay, Walmart, Trustpilot, and more — with a universal fallback scraper for any other website.

---

## Key Features

| Feature | Description |
|---|---|
| 🔗 **Bulk URL Scan** | Paste any product URL to automatically scrape and classify all reviews on the page |
| 📝 **Single Review Analysis** | Paste a single review text for instant fake/genuine classification |
| 🧠 **ML Classification** | Logistic Regression model trained on labelled review data with TF-IDF vectorization |
| 🔍 **Linguistic Heuristic Engine** | Detects exclamation overuse, ALL-CAPS, superlative overload, template phrases, repetition, and emotional intensity |
| 📐 **Structural Analysis** | Flags run-on sentences, punctuation anomalies, emoji overuse, and formatting irregularities |
| ✨ **Gemini AI Cross-Verification** | Optional integration with Google Gemini 2.0 Flash for semantic authenticity evaluation |
| 🌐 **Playwright JS Rendering** | Headless Chromium fallback for JavaScript-heavy sites (Flipkart, Meesho, Myntra, etc.) |
| 🕵️ **Aggregate Pattern Detection** | Identifies coordinated review campaigns via similarity clustering, length uniformity, and trust score analysis |
| 📊 **Interactive Dashboard** | Donut charts, probability bars, trust scores, filter controls, and per-review breakdowns |
| 🔐 **Secure API Key Management** | Gemini API key stored only in-memory, cleared on server restart |

---

## How It Works

ReviewGuard uses a **6-stage analysis pipeline**:

```
┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐    ┌────────────────┐    ┌──────────────┐
│  SCRAPE  │ →  │  PREPROCESS  │ →  │ VECTORIZE │ →  │ CLASSIFY │ →  │ HEURISTIC CHECK│ →  │ AI CROSS-CHECK│
│          │    │              │    │           │    │          │    │                │    │   (Optional)  │
│ Static + │    │ Lowercase,   │    │ TF-IDF    │    │ Logistic │    │ Linguistic +   │    │ Gemini 2.0   │
│ Playwright│   │ Lemmatize,   │    │ bigrams,  │    │ Regression│   │ Structural     │    │ Flash        │
│ fallback │    │ Stop words   │    │ 50K feat  │    │          │    │ scoring        │    │              │
└──────────┘    └──────────────┘    └───────────┘    └──────────┘    └────────────────┘    └──────────────┘
```

1. **Scrape** — Multi-strategy scraper extracts reviews via JSON parsing, schema.org metadata, keyword-based DOM search, or paragraph extraction. Falls back to Playwright headless Chromium for JS-rendered pages.
2. **Preprocess** — Lowercasing, HTML stripping, URL removal, digit removal, punctuation cleanup, stopword filtering, and POS-aware WordNet lemmatization.
3. **Vectorize** — TF-IDF with unigrams + bigrams, 50,000-feature sparse matrix, sublinear TF scaling.
4. **Classify** — Logistic Regression model predicts Fake (0) or Genuine (1) with probability scores.
5. **Heuristic Check** — 8+ linguistic rules and 5+ structural rules compute a suspicion score (0–100) and flag anomalies.
6. **AI Cross-Check** — Gemini 2.0 Flash evaluates each review semantically and provides an independent authenticity verdict with confidence and reasoning.

The final **Trust Score** blends ML confidence, heuristic suspicion, and (if enabled) Gemini AI confidence using a weighted formula: **60% ML+Heuristic, 40% Gemini AI**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, Flask |
| **ML/NLP** | scikit-learn (Logistic Regression, TF-IDF), NLTK (WordNet lemmatizer, POS tagger, stopwords) |
| **Web Scraping** | Requests, BeautifulSoup4, lxml |
| **JS Rendering** | Playwright (headless Chromium) |
| **AI Layer** | Google Gemini 2.0 Flash (`google-genai` SDK) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Data** | Kaggle fake reviews dataset / synthetic fallback |
| **Serialization** | joblib (model persistence) |

---

## Project Structure

```
ML_MDOULE5_PROJECT/
│
├── app.py                  # Flask application — routes, scraping engine, ML inference, Gemini AI
├── train_model.py          # Model training pipeline — data loading, preprocessing, training, evaluation
├── test_urls.py            # URL scraping test script
├── requirements.txt        # Python dependencies
├── .gitignore
│
├── data/
│   └── reviews.csv         # Training dataset (Kaggle or synthetic)
│
├── model_artifacts/
│   ├── tfidf_vectorizer.pkl    # Trained TF-IDF vectorizer
│   └── classifier.pkl          # Trained Logistic Regression model
│
├── templates/
│   └── index.html          # Main frontend HTML
│
├── static/
│   ├── style.css           # Stylesheet
│   └── script.js           # Frontend JavaScript (API calls, UI logic, charts)
│
└── venv/                   # Python virtual environment (not committed)
```

---

## Getting Started

### Prerequisites

- **Python 3.10+**
- **pip** (Python package manager)
- (Optional) **Kaggle API credentials** at `~/.kaggle/kaggle.json` for automatic dataset download
- (Optional) **Google Gemini API key** from [aistudio.google.com](https://aistudio.google.com) for AI cross-verification

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ML_MDOULE5_PROJECT.git
   cd ML_MDOULE5_PROJECT
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install Playwright for JS-rendered sites:**
   ```bash
   pip install playwright
   playwright install chromium
   ```

### Train the Model

Before running the app, you need to train the ML model:

```bash
python train_model.py
```

This will:
- Attempt to download the [Kaggle Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) (requires `~/.kaggle/kaggle.json`)
- Fall back to a local `data/reviews.csv` if Kaggle credentials are unavailable
- Generate a **synthetic dataset** (600 reviews) as a last resort
- Train a **TF-IDF + Logistic Regression** pipeline
- Save model artifacts to `model_artifacts/`
- Print accuracy, 5-fold cross-validation scores, and a classification report

### Run the Application

```bash
python app.py
```

The server starts at **http://localhost:5000**. Open this URL in your browser.

---

## Usage

### URL Scan Mode

1. Select the **🔗 Product Link** tab
2. Paste any product URL (e.g., `https://www.amazon.in/dp/B0XXXXXX`)
3. Click **Scan Reviews**
4. ReviewGuard will scrape, classify, and display:
   - Overall genuine vs. fake breakdown (donut chart)
   - Individual review verdicts with confidence scores
   - Trust scores and heuristic flags per review
   - Aggregate suspicious patterns (copy-paste detection, length uniformity, campaign alerts)

### Single Review Mode

1. Select the **📝 Paste Review** tab
2. Paste or type a review text
3. Click **Analyse** (or press `Ctrl+Enter`)
4. View the verdict, confidence meter, probability bars, trust score, and heuristic signals

### Gemini AI Setup (Optional)

1. Click the **⚙️ Settings** icon in the header
2. Enter your Gemini API key (get one free at [aistudio.google.com](https://aistudio.google.com))
3. Click **Save**
4. The **🤖 AI** badge turns green — Gemini will now cross-verify every review

> **Note:** The API key is stored only in server memory and is automatically cleared when the server restarts. It is never written to disk.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the main web interface |
| `GET` | `/health` | Health check — returns model load status |
| `POST` | `/predict` | Classify a single review text |
| `POST` | `/analyse-url` | Scrape and classify all reviews from a URL |
| `POST` | `/set-api-key` | Set or clear the Gemini API key (in-memory) |
| `GET` | `/gemini-status` | Check Gemini and Playwright availability |

### Example: `/predict`

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text": "This product is absolutely the BEST EVER!!!"}'
```

**Response:**
```json
{
  "prediction": "Fake",
  "confidence_pct": 92.5,
  "fake_prob": 92.5,
  "genuine_prob": 7.5,
  "trust_score": 15.3,
  "verification_status": "high_confidence_fake",
  "flags": ["Excessive exclamation marks (3)", "Multiple ALL-CAPS (2 words)"],
  "analysis": { ... }
}
```

### Example: `/analyse-url`

```bash
curl -X POST http://localhost:5000/analyse-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.amazon.in/dp/B0EXAMPLE"}'
```

---

## Supported Platforms

ReviewGuard includes **dedicated scrapers** for these platforms, plus a **universal smart scraper** fallback for any other website:

| Platform | Region | Scraper Type |
|---|---|---|
| Amazon (.com, .in, .co.uk, etc.) | Global | Dedicated |
| Flipkart | India | Dedicated |
| Meesho | India | Smart |
| Myntra | India | Smart |
| Snapdeal | India | Smart |
| eBay | Global | Dedicated |
| Walmart | US | Dedicated |
| Best Buy | US | Smart |
| Target | US | Smart |
| Costco | US | Smart |
| Newegg | US | Smart |
| Etsy | Global | Smart |
| AliExpress | Global | Smart |
| Alibaba | Global | Smart |
| Temu | Global | Smart |
| SHEIN | Global | Smart |
| JD.com | China | Smart |
| Lazada | SEA | Smart |
| Shopee | SEA | Smart |
| Noon | Middle East | Smart |
| Jumia | Africa | Smart |
| Trustpilot | Global | Dedicated |
| Yelp | US | Smart |
| TripAdvisor | Global | Dedicated |
| Google Reviews | Global | Smart |
| **Any other site** | — | Universal fallback |

---

## Screenshots

After running the app, visit `http://localhost:5000` to see:

- **Hero Section** — Modern UI with ML / NLP / AI badges
- **URL Scan Dashboard** — Donut charts, probability bars, individual review cards with trust scores
- **Single Review Analyser** — Radial confidence meter, heuristic signal chips
- **Settings Panel** — Gemini API key management with live status indicators

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>🛡️ ReviewGuard</strong> — Powered by scikit-learn · NLTK · Flask · BeautifulSoup · Gemini AI · Playwright
</p>
]]>
