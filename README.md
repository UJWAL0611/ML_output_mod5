# render link:https://ml-output-mod5-1.onrender.com/
<p align="center">
  <img src="https://img.shields.io/badge/🛡️_ReviewGuard-Universal_Fake_Review_Detector-blueviolet?style=for-the-badge&labelColor=1a1a2e" alt="ReviewGuard" />
</p>

<h1 align="center">🛡️ ReviewGuard</h1>
<h3 align="center">Universal Fake Review Detector</h3>

<p align="center">
  Detect fake product reviews instantly using a multi-layered intelligence engine<br/>
  powered by <b>Machine Learning</b>, <b>NLP Heuristics</b>, and <b>Google Gemini AI</b>.
</p>

<p align="center">
  <a href="https://github.com/UJWAL0611/ML_output_mod5"><img src="https://img.shields.io/badge/📦_Repository-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Repo" /></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-3.1-000000?style=flat-square&logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat-square&logo=scikitlearn&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-3.9-154F5B?style=flat-square" />
  <img src="https://img.shields.io/badge/Gemini_AI-2.0_Flash-4285F4?style=flat-square&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/Playwright-45ba4b?style=flat-square&logo=playwright&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Platforms-26+-orange?style=flat-square" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Frontend](#-frontend)
- [Backend](#-backend)
- [Machine Learning Engine](#-machine-learning-engine)
- [NLP Preprocessing Pipeline](#-nlp-preprocessing-pipeline)
- [Web Scraping Engine](#-web-scraping-engine)
- [Gemini AI Intelligence Layer](#-gemini-ai-intelligence-layer)
- [Heuristic Analysis Module](#-heuristic-analysis-module)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [API Endpoints](#-api-endpoints)
- [Supported Platforms](#-supported-platforms)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔭 Overview

**ReviewGuard** is a full-stack web application that identifies fake and deceptive product reviews across any e-commerce platform. Online shopping is heavily influenced by user reviews, but a significant portion of these reviews are fabricated — either by sellers boosting their own products or by competitors posting negative reviews. ReviewGuard tackles this problem by combining multiple layers of analysis: a trained **Machine Learning classifier**, **rule-based linguistic heuristics**, and optional **Google Gemini AI cross-verification** — all delivered through a modern, interactive web dashboard.

The system supports **26+ e-commerce platforms** out of the box — including Amazon, Flipkart, Meesho, Myntra, eBay, Walmart, Trustpilot, and more — with a universal fallback scraper for any other website. Users can either paste a product URL for bulk review scanning or paste individual review text for single analysis.

---

## 🎨 Frontend

The frontend is built with **vanilla HTML5, CSS3, and JavaScript** — no frameworks, no build tools, zero dependencies. The UI follows a dark-mode-first design philosophy with a premium glassmorphism aesthetic, built using custom CSS variables for a consistent design system.

### Design System

The interface uses the **Syne** typeface for headings and **DM Mono** for body/code text, loaded from Google Fonts. The color palette is carefully curated around a dark base (`#0a0c0f`) with an accent blue (`#38bdf8`), semantic green for genuine reviews (`#34d399`), and semantic red for fake reviews (`#f43f5e`). A subtle CSS grid background with a radial gradient glow creates depth without distraction.

### UI Components

- **Sticky Header** — Frosted-glass header with backdrop blur, showing the ReviewGuard logo, ML/NLP/AI status badges, and a settings gear icon.
- **Tab Switcher** — Toggle between "Product Link" (bulk URL scan) and "Paste Review" (single text analysis) modes with animated tab pills.
- **URL Scan Dashboard** — Features a URL input bar with auto-prefix, a loading spinner with status text, an animated SVG donut chart showing genuine vs. fake breakdown, horizontal probability bars, a verdict chip with aggregate trust score, filter buttons (All / Fake / Genuine), and a scrollable review list with per-review trust badges, heuristic flag chips, and AI verdict indicators.
- **Single Review Analyzer** — Split-pane layout with a textarea input (character counter, Ctrl+Enter shortcut) on the left and a results panel on the right, featuring a radial SVG confidence meter, probability bars, trust score box, heuristic signal chips, and Gemini AI verdict display.
- **Settings Panel** — Slide-down modal overlay for Gemini API key management with password input, save/clear buttons, and live status dots for Gemini library, API key, and Playwright availability.
- **Suspicious Patterns Box** — Gradient-bordered intelligence section that surfaces coordinated manipulation patterns like copy-paste reviews, uniform lengths, and campaign alerts.

### Responsiveness

The CSS includes full responsive breakpoints at `800px` and `480px`. On mobile, the split-pane card switches to a stacked layout, the step grid collapses to single columns, and padding/font sizes adjust for touch targets.

### Key Files

| File | Description |
|---|---|
| `templates/index.html` | 394 lines — Semantic HTML5 with ARIA attributes, SVG charts, and structured sections |
| `static/style.css` | 753 lines — Complete design system with CSS variables, animations, glassmorphism, and responsive breakpoints |
| `static/script.js` | 596 lines — All client-side logic: API calls, DOM manipulation, chart rendering, tab switching, settings management |

---

## 🖥️ Backend

The backend is a **Flask** web server (`app.py` — 1,180 lines) that serves the frontend, exposes REST API endpoints, orchestrates the scraping-classification pipeline, and manages the Gemini AI integration. It runs on `0.0.0.0:5000` by default.

### Architecture

The backend is organized into clearly separated modules within a single file:

1. **NLP Preprocessing** (lines 47–89) — NLTK setup, WordNet lemmatizer, stopword filtering, and the `preprocess()` function.
2. **Model Loading** (lines 92–100) — Loads the pre-trained TF-IDF vectorizer and Logistic Regression classifier from `model_artifacts/` using joblib.
3. **Intelligence Module** (lines 103–369) — Multi-layer verification engine with linguistic scoring, structural scoring, cross-verification logic, and the enhanced `_predict_single()` function.
4. **Scraping Engine** (lines 432–805) — Platform-specific scrapers, universal smart scraper, Playwright JS-rendering fallback, and the main `scrape_reviews()` orchestrator.
5. **Gemini AI Layer** (lines 808–869) — Batch review verification using the Google `genai` SDK with structured JSON prompt engineering.
6. **Flask Routes** (lines 875–1178) — Six endpoints for the web interface, health check, prediction, URL analysis, API key management, and status checking.

### Configuration

| Setting | Value | Description |
|---|---|---|
| `MAX_CONTENT_LENGTH` | 10 MB | Maximum request body size |
| `MAX_REVIEW_CHARS` | 10,000 | Maximum characters per single review |
| `MAX_URL_LENGTH` | 4,096 | Maximum URL length accepted |
| `SCRAPE_TIMEOUT` | 20 sec | HTTP request timeout for scraping |
| `MAX_REVIEWS` | 50 | Maximum reviews extracted per URL |

### Security

- Gemini API keys are stored **only in server memory** using a thread-safe lock (`threading.Lock`), never written to disk, and cleared on restart.
- SSL warnings are suppressed for scraping flexibility (`urllib3.disable_warnings`).
- Input validation on all endpoints with character limits and URL format checks.
- `MAX_CONTENT_LENGTH` prevents oversized payloads.

---

## 🧠 Machine Learning Engine

The ML engine is a **Logistic Regression classifier** trained on TF-IDF vectorized review text. The training pipeline is implemented in `train_model.py` (309 lines).

### Training Pipeline

1. **Data Loading** — Tries three sources in order:
   - **Kaggle**: Downloads the [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) by Mexwell (requires `~/.kaggle/kaggle.json`). Maps `CG` → Fake (0), `OR` → Genuine (1).
   - **Local CSV**: Reads `data/reviews.csv` with columns `review_text` and `label`.
   - **Synthetic Fallback**: Generates 600 reviews (300 fake + 300 genuine) from templates with augmentation words.

2. **Preprocessing** — Applies the full NLP pipeline (see next section) to every review.

3. **Vectorization** — TF-IDF with:
   - Unigrams + bigrams (`ngram_range=(1, 2)`)
   - 50,000 max features
   - Sublinear TF scaling
   - Minimum document frequency of 2
   - Unicode accent stripping

4. **Classification** — Logistic Regression with:
   - Regularization `C=5.0`
   - `lbfgs` solver, 1000 max iterations
   - Balanced class weights (handles imbalanced data)

5. **Evaluation** — 80/20 stratified train/test split + 5-fold stratified cross-validation. Outputs accuracy, per-fold scores, and a full classification report (precision, recall, F1).

6. **Serialization** — Saves `tfidf_vectorizer.pkl` and `classifier.pkl` to `model_artifacts/` using joblib with compression level 3.

### Inference

At runtime, `_predict_single()` preprocesses the input text, transforms it through the loaded TF-IDF vectorizer, and runs `predict_proba()` to get Fake/Genuine probabilities. The output includes the predicted class, confidence percentage, and both class probabilities — which are then passed to the cross-verification engine.

---

## 📝 NLP Preprocessing Pipeline

Every review — whether from the training set or user input — passes through an 8-step NLP pipeline built on **NLTK**:

| Step | Operation | Example |
|:---:|---|---|
| 1 | **Lowercase** | `"BEST Product"` → `"best product"` |
| 2 | **Strip HTML tags** | `"<b>great</b>"` → `"great"` |
| 3 | **Strip URLs** | `"check https://x.com"` → `"check"` |
| 4 | **Remove digits** | `"model 2024"` → `"model"` |
| 5 | **Remove punctuation** | `"amazing!!!"` → `"amazing"` |
| 6 | **Remove stopwords** | `"this is a great"` → `"great"` |
| 7 | **POS-aware lemmatization** | `"running better"` → `"run well"` |
| 8 | **Drop short tokens** | Removes tokens with fewer than 3 characters |

The lemmatizer uses **WordNet** with POS tag mapping — adjectives, verbs, and adverbs are lemmatized with their correct part of speech (e.g., `"better"` → `"well"` as adverb, not `"better"` as noun). This produces higher-quality features than stemming alone.

### NLTK Resources Downloaded

`stopwords`, `wordnet`, `averaged_perceptron_tagger`, `averaged_perceptron_tagger_eng`, `punkt`, `punkt_tab` — all downloaded automatically with SSL bypass for Windows compatibility.

---

## 🕷️ Web Scraping Engine

The scraping engine extracts reviews from any website using a **4-strategy cascade** with platform-specific optimizations and a JS-rendering fallback.

### Extraction Strategies (in priority order)

1. **JSON Extraction** — Parses embedded JSON blobs in page source, searching for keys like `reviewText`, `reviewBody`, `content`, `comment`, etc.
2. **Schema.org Metadata** — Finds elements with `itemprop="reviewBody"` or `itemprop="description"`.
3. **Keyword-based DOM Search** — Hunts for elements with class/id/data attributes matching patterns like `review`, `comment`, `feedback`, `testimonial`, etc.
4. **Paragraph Fallback** — Last resort extraction of all `<p>` and `<li>` elements with 8–300 words.

### Anti-Detection

The scraper rotates through **4 User-Agent profiles** (Chrome/Windows, Safari/Mac, Firefox/Windows, Chrome/Android) with full browser-like headers including `Sec-Fetch-*`, `Sec-Ch-Ua-*`, and `Cache-Control` headers. It uses a `requests.Session` with retry adapters and SSL verification disabled for maximum compatibility.

### Platform-Specific Scrapers

Dedicated scrapers with platform-specific logic for **Amazon** (ASIN extraction, `data-hook="review-body"`), **Flipkart** (PID extraction, reviews tab URL), **eBay**, **Walmart**, **Trustpilot** (`data-service-review-text-typography`), and **TripAdvisor** (class-based extraction).

### Playwright JS-Rendering Fallback

When static scraping returns zero reviews, the engine automatically falls back to **Playwright headless Chromium**. It launches a browser context with a realistic viewport (1280×900), navigates to the page, auto-scrolls 5 times to trigger lazy-loading, then extracts the fully rendered HTML for parsing. This handles JavaScript-heavy sites like Flipkart, Meesho, and Myntra.

### Post-Processing

All extracted reviews pass through `_clean()` which deduplicates (by first 120 chars), filters by word count (6–500 words), and caps at 50 reviews.

---

## ✨ Gemini AI Intelligence Layer

The optional **Google Gemini 2.0 Flash** integration provides a semantic cross-verification layer on top of the ML classifier and heuristic engine.

### How It Works

1. Reviews are batched (up to 15 at a time) and sent to the Gemini API with a structured prompt requesting a JSON array response.
2. Each review gets an independent AI verdict: `authentic` (true/false), `confidence` (0–100), and a short `reason` (max 12 words).
3. The AI verdict is blended with the ML+heuristic trust score using a **60/40 weighted formula**:
   ```
   Final Trust = (ML+Heuristic Trust × 0.60) + (Gemini AI Trust × 0.40)
   ```
4. If the ML model and Gemini AI **strongly disagree** (confidence ≥ 70%), the review is flagged with an `ai_conflict` status, displayed as a special "⚡ AI Conflict" chip in the UI.

### Setup

The Gemini API key is set via the Settings panel in the UI or the `/set-api-key` endpoint. It is stored only in-memory with thread-safe access and never persisted to disk. Get a free API key at [aistudio.google.com](https://aistudio.google.com).

---

## 🔍 Heuristic Analysis Module

Beyond ML classification, ReviewGuard applies **two layers of rule-based heuristic analysis** to every review, computing a suspicion score (0–100) and flagging specific patterns.

### Linguistic Analysis (8 rules)

| Rule | What It Detects | Score Impact |
|---|---|---|
| Exclamation overuse | 3+ or 5+ exclamation marks | +10 or +20 |
| ALL-CAPS words | 3+ or 5+ words in full caps | +10 or +20 |
| Superlative overload | 2+ or 4+ phrases like "best ever", "changed my life" | +12 or +25 |
| Short review | Less than 10 or 15 words | +15 or +8 |
| Word repetition | Unique word ratio < 50% in 20+ word reviews | +15 |
| Template language | 2+ phrases like "honest review", "fast shipping" | +18 |
| Lack of specificity | No numbers or measurements in 30+ word reviews | +12 |
| Emotional intensity | 4+ emotional words (love, hate, amazing, etc.) | +10 |

### Structural Analysis (5 rules)

| Rule | What It Detects | Score Impact |
|---|---|---|
| Run-on sentence | Single sentence with 30+ words | +15 |
| Excessive ellipsis | 3+ instances of "..." | +10 |
| Repeated punctuation | `??` or `!!` patterns | +12 |
| Long unpunctuated text | 40+ words with ≤2 sentences | +15 |
| Emoji overuse | 3+ or 5+ emojis | +8 or +15 |

### Aggregate Pattern Detection

For URL bulk scans, an additional layer analyzes patterns **across** all reviews:

- **Similarity Clustering** — Detects copy-paste reviews with >70% word overlap
- **Length Uniformity** — Flags suspiciously uniform review lengths (σ < 5)
- **Fake Ratio Analysis** — Alerts when 60%+ or 80%+ reviews are classified fake
- **Trust Score Clustering** — Detects uniform trust scores (σ < 8) suggesting bot-generated content
- **Flag Concentration** — Identifies when 40%+ reviews share the same heuristic flags

---

## 🛠️ Tech Stack

<table>
  <tr>
    <th align="left">Layer</th>
    <th align="left">Technologies</th>
  </tr>
  <tr>
    <td>🎨 Frontend</td>
    <td><img src="https://img.shields.io/badge/HTML5-E34F26?style=flat-square&logo=html5&logoColor=white"/> <img src="https://img.shields.io/badge/CSS3-1572B6?style=flat-square&logo=css3&logoColor=white"/> <img src="https://img.shields.io/badge/JavaScript-F7DF1E?style=flat-square&logo=javascript&logoColor=black"/></td>
  </tr>
  <tr>
    <td>🖥️ Backend</td>
    <td><img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=flat-square&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Flask_3.1-000?style=flat-square&logo=flask&logoColor=white"/> <img src="https://img.shields.io/badge/Gunicorn-499848?style=flat-square&logo=gunicorn&logoColor=white"/></td>
  </tr>
  <tr>
    <td>🧠 ML / NLP</td>
    <td><img src="https://img.shields.io/badge/scikit--learn_1.8-F7931E?style=flat-square&logo=scikitlearn&logoColor=white"/> <img src="https://img.shields.io/badge/NLTK_3.9-154F5B?style=flat-square"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"/></td>
  </tr>
  <tr>
    <td>🕷️ Scraping</td>
    <td><img src="https://img.shields.io/badge/BeautifulSoup4-555?style=flat-square"/> <img src="https://img.shields.io/badge/Requests-red?style=flat-square"/> <img src="https://img.shields.io/badge/lxml-green?style=flat-square"/></td>
  </tr>
  <tr>
    <td>🌐 JS Rendering</td>
    <td><img src="https://img.shields.io/badge/Playwright-45ba4b?style=flat-square&logo=playwright&logoColor=white"/></td>
  </tr>
  <tr>
    <td>🤖 AI Layer</td>
    <td><img src="https://img.shields.io/badge/Google_Gemini_2.0_Flash-4285F4?style=flat-square&logo=google&logoColor=white"/></td>
  </tr>
  <tr>
    <td>💾 Serialization</td>
    <td><img src="https://img.shields.io/badge/joblib-888?style=flat-square"/></td>
  </tr>
</table>

---

## 📂 Project Structure

```
ML_MDOULE5_PROJECT/
│
├── 📄 app.py                     # Flask server — routes, scraping, ML inference, Gemini AI (1180 lines)
├── 📄 train_model.py             # Training pipeline — data loading, preprocessing, evaluation (309 lines)
├── 📄 test_urls.py               # URL scraping test script
├── 📄 requirements.txt           # Python dependencies (13 packages)
├── 📄 .gitignore                 # Git exclusions
├── 📄 README.md                  # This file
│
├── 📁 data/
│   └── reviews.csv               # Training dataset — Kaggle download or synthetic (600+ rows)
│
├── 📁 model_artifacts/
│   ├── tfidf_vectorizer.pkl      # Trained TF-IDF vectorizer (50K features, bigrams)
│   └── classifier.pkl            # Trained Logistic Regression classifier
│
├── 📁 templates/
│   └── index.html                # Frontend HTML template (394 lines)
│
└── 📁 static/
    ├── style.css                 # Dark-mode design system with animations (753 lines)
    └── script.js                 # Client-side logic — API calls, charts, UI (596 lines)
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** and **pip**
- _(Optional)_ Kaggle credentials at `~/.kaggle/kaggle.json` for real dataset
- _(Optional)_ [Google Gemini API key](https://aistudio.google.com) for AI cross-verification

### 1️⃣ Clone

```bash
git clone https://github.com/UJWAL0611/ML_output_mod5.git
cd ML_output_mod5
```

### 2️⃣ Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ (Optional) Playwright for JS Sites

```bash
pip install playwright
playwright install chromium
```

### 5️⃣ Train the Model

```bash
python train_model.py
```

This trains the TF-IDF + Logistic Regression pipeline, runs 5-fold cross-validation, prints a classification report, and saves model artifacts.

### 6️⃣ Run the App

```bash
python app.py
```

Open **http://localhost:5000** in your browser 🚀

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|:---:|---|---|
| `GET` | `/` | Serve the web interface |
| `GET` | `/health` | Health check — model load status |
| `POST` | `/predict` | Classify a single review text |
| `POST` | `/analyse-url` | Scrape + classify all reviews from a URL |
| `POST` | `/set-api-key` | Set/clear the Gemini API key (in-memory only) |
| `GET` | `/gemini-status` | Check Gemini and Playwright availability |

<details>
<summary><b>📬 Example: POST /predict</b></summary>

**Request:**
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
  "analysis": {
    "word_count": 8,
    "unique_ratio": 0.88,
    "sentence_count": 1,
    "linguistic_score": 45,
    "structural_score": 12
  }
}
```
</details>

<details>
<summary><b>📬 Example: POST /analyse-url</b></summary>

**Request:**
```bash
curl -X POST http://localhost:5000/analyse-url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.amazon.in/dp/B0EXAMPLE"}'
```

**Response includes:** `total`, `genuine_count`, `fake_count`, `genuine_pct`, `fake_pct`, `reviews[]` (per-review analysis with trust scores, flags, and AI verdicts), `aggregate_trust`, `suspicious_patterns[]`, `pattern_flags`, `gemini_active`, and `playwright_active`.
</details>

---

## 🌍 Supported Platforms

<table>
  <tr>
    <th>🇮🇳 India</th>
    <th>🇺🇸 United States</th>
    <th>🌎 Global</th>
    <th>🌏 Asia & Other</th>
  </tr>
  <tr><td>Amazon.in ⭐</td><td>Amazon.com ⭐</td><td>eBay ⭐</td><td>JD.com</td></tr>
  <tr><td>Flipkart ⭐</td><td>Walmart ⭐</td><td>AliExpress</td><td>Lazada</td></tr>
  <tr><td>Meesho</td><td>Best Buy</td><td>Alibaba</td><td>Shopee</td></tr>
  <tr><td>Myntra</td><td>Target</td><td>Etsy</td><td>Noon (Middle East)</td></tr>
  <tr><td>Snapdeal</td><td>Costco</td><td>Trustpilot ⭐</td><td>Jumia (Africa)</td></tr>
  <tr><td></td><td>Newegg</td><td>TripAdvisor ⭐</td><td>Temu</td></tr>
  <tr><td></td><td>Yelp</td><td>Google Reviews</td><td>SHEIN</td></tr>
</table>

<p align="center"><sub>⭐ = Dedicated platform-specific scraper &nbsp;|&nbsp; All others use the universal smart scraper<br/>+ <b>any other website</b> via the universal fallback scraper</sub></p>

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "Add my feature"`
4. **Push** to the branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>🛡️ ReviewGuard</b><br/>
  <sub>Powered by scikit-learn · NLTK · Flask · BeautifulSoup · Gemini AI · Playwright</sub>
</p>

<p align="center">
  <sub>Made with ❤️ by <a href="https://github.com/UJWAL0611">Ujwal A Bhansali</a></sub>
</p>
