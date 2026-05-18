// script.js
"use strict";

const CIRCUMFERENCE = 339.29; // 2 * Math.PI * 54
const MAX_CHARS     = 10000;
const WARN_CHARS    = 9000;

// ── Tab switching ─────────────────────────────────────────────────
const tabBtns     = document.querySelectorAll(".tab-btn");
const sectionUrl  = document.getElementById("section-url");
const sectionText = document.getElementById("section-text");

tabBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    if (tab === "url") {
      sectionUrl.classList.remove("hidden");
      sectionText.classList.add("hidden");
    } else {
      sectionText.classList.remove("hidden");
      sectionUrl.classList.add("hidden");
    }
  });
});

// ══════════════════════════════════════════════════════════════════
// SECTION 1 — PRODUCT URL SCAN
// ══════════════════════════════════════════════════════════════════
const urlInput      = document.getElementById("url-input");
const btnScanUrl    = document.getElementById("btn-scan-url");
const btnClearUrl   = document.getElementById("btn-clear-url");

const urlIdle       = document.getElementById("url-idle");
const urlLoading    = document.getElementById("url-loading");
const urlResultBody = document.getElementById("url-result-body");
const urlError      = document.getElementById("url-error");
const urlErrorMsg   = document.getElementById("url-error-msg");

const urlTotal          = document.getElementById("url-total");
const urlGenuineCount   = document.getElementById("url-genuine-count");
const urlFakeCount      = document.getElementById("url-fake-count");
const urlBarGenuine     = document.getElementById("url-bar-genuine");
const urlBarFake        = document.getElementById("url-bar-fake");
const urlPctGenuine     = document.getElementById("url-pct-genuine");
const urlPctFake        = document.getElementById("url-pct-fake");
const urlVerdictChip    = document.getElementById("url-verdict-chip");
const donutGenuineArc   = document.getElementById("donut-genuine-arc");
const donutFakeArc      = document.getElementById("donut-fake-arc");
const donutLabel        = document.getElementById("donut-label");
const reviewList        = document.getElementById("review-list");
const filterBtns        = document.querySelectorAll(".filter-btn");

let allReviews = [];   // store for filter

btnClearUrl.addEventListener("click", () => {
  urlInput.value = "";
  showUrlState(urlIdle);
  allReviews = [];
});

btnScanUrl.addEventListener("click", runUrlScan);
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runUrlScan();
});

async function runUrlScan() {
  const url = urlInput.value.trim();
  if (!url) { showUrlError("Please enter a product URL."); return; }
  if (!/^https?:\/\//i.test(url)) { showUrlError("URL must start with http:// or https://"); return; }

  setUrlLoading(true);
  showUrlState(urlLoading);

  try {
    const resp = await fetch("/analyse-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await resp.json();

    if (!resp.ok) {
      showUrlError(data.error || `Server error (HTTP ${resp.status}).`);
      return;
    }
    renderUrlResult(data);
  } catch (err) {
    showUrlError("Network error — could not reach the server.");
  } finally {
    setUrlLoading(false);
  }
}

function renderUrlResult(data) {
  allReviews = data.reviews || [];

  // Counts
  urlTotal.textContent        = data.total;
  urlGenuineCount.textContent = data.genuine_count;
  urlFakeCount.textContent    = data.fake_count;

  // Bars
  urlBarGenuine.style.width = `${data.genuine_pct}%`;
  urlBarFake.style.width    = `${data.fake_pct}%`;
  urlPctGenuine.textContent = `${data.genuine_pct}%`;
  urlPctFake.textContent    = `${data.fake_pct}%`;

  // Donut chart — two-segment arc
  animateDonut(data.genuine_pct, data.fake_pct);

  // Donut centre label
  const dominant = data.genuine_pct >= data.fake_pct ? "genuine" : "fake";
  donutLabel.textContent  = dominant === "genuine" ? `${data.genuine_pct}%` : `${data.fake_pct}%`;
  donutLabel.style.color  = dominant === "genuine"
    ? getComputedStyle(document.documentElement).getPropertyValue("--genuine").trim()
    : getComputedStyle(document.documentElement).getPropertyValue("--fake").trim();

  // Verdict chip
  urlVerdictChip.className = "url-verdict-chip";
  if (data.genuine_pct >= 70) {
    urlVerdictChip.classList.add("mostly-genuine");
    urlVerdictChip.textContent = "✅ Mostly Genuine";
  } else if (data.fake_pct >= 70) {
    urlVerdictChip.classList.add("mostly-fake");
    urlVerdictChip.textContent = "🚨 Mostly Fake";
  } else {
    urlVerdictChip.classList.add("mixed");
    urlVerdictChip.textContent = "⚖️ Mixed Reviews";
  }

  // Render review list
  renderReviewList(allReviews);
  showUrlState(urlResultBody);
}

function animateDonut(genuinePct, fakePct) {
  const C = 339.29;
  const genuineLen = C * (genuinePct / 100);
  const fakeLen    = C * (fakePct / 100);

  // Genuine arc starts at 0 (top after -90deg rotation)
  donutGenuineArc.style.strokeDasharray  = `${genuineLen} ${C}`;
  donutGenuineArc.style.strokeDashoffset = "0";

  // Fake arc starts right after genuine
  donutFakeArc.style.strokeDasharray  = `${fakeLen} ${C}`;
  donutFakeArc.style.strokeDashoffset = `${-genuineLen}`;
}

// ── Review list rendering ─────────────────────────────────────────
function renderReviewList(reviews) {
  reviewList.innerHTML = "";
  if (!reviews.length) {
    reviewList.innerHTML = '<p style="padding:20px 32px;color:var(--text-dim);font-size:0.85rem;">No reviews to display.</p>';
    return;
  }
  reviews.forEach((r, idx) => {
    const item = document.createElement("div");
    item.className = `review-item ${r.prediction === "Fake" ? "is-fake" : "is-genuine"}`;
    item.dataset.prediction = r.prediction;

    const words = r.text.trim().split(/\s+/).length;
    const needsExpand = words > 40;

    item.innerHTML = `
      <div class="review-text-col">
        <p class="review-text-body" id="rtb-${idx}">${escapeHtml(r.text)}</p>
        ${needsExpand ? `<button class="review-expand-btn" data-idx="${idx}" type="button">Show more</button>` : ""}
        <span class="review-meta">${r.confidence_pct}% confidence · Fake ${r.fake_prob}% · Genuine ${r.genuine_prob}%</span>
      </div>
      <span class="review-badge ${r.prediction === "Fake" ? "fake" : "genuine"}">
        ${r.prediction === "Fake" ? "🚨 Fake" : "✅ Genuine"}
      </span>`;
    reviewList.appendChild(item);
  });

  // Expand buttons
  reviewList.querySelectorAll(".review-expand-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const idx = btn.dataset.idx;
      const p = document.getElementById(`rtb-${idx}`);
      if (p.classList.contains("expanded")) {
        p.classList.remove("expanded");
        btn.textContent = "Show more";
      } else {
        p.classList.add("expanded");
        btn.textContent = "Show less";
      }
    });
  });
}

function escapeHtml(str) {
  return str.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── Filter buttons ────────────────────────────────────────────────
filterBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    filterBtns.forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const f = btn.dataset.filter;
    const filtered = f === "all" ? allReviews : allReviews.filter((r) => r.prediction === f);
    renderReviewList(filtered);
  });
});

// ── URL state helpers ─────────────────────────────────────────────
function showUrlState(activeEl) {
  [urlIdle, urlLoading, urlResultBody, urlError].forEach((el) => {
    el === activeEl ? el.classList.remove("hidden") : el.classList.add("hidden");
  });
}
function showUrlError(msg) {
  urlErrorMsg.textContent = msg;
  showUrlState(urlError);
}
function setUrlLoading(on) {
  btnScanUrl.disabled = on;
  btnClearUrl.disabled = on;
  urlInput.disabled = on;
  on ? btnScanUrl.classList.add("loading") : btnScanUrl.classList.remove("loading");
}


// ══════════════════════════════════════════════════════════════════
// SECTION 2 — SINGLE REVIEW PASTE
// ══════════════════════════════════════════════════════════════════
const reviewInput   = document.getElementById("review-input");
const charCount     = document.getElementById("char-count");
const btnClear      = document.getElementById("btn-clear");
const btnAnalyse    = document.getElementById("btn-analyse");
const stateIdle     = document.getElementById("state-idle");
const stateResult   = document.getElementById("state-result");
const stateError    = document.getElementById("state-error");
const verdictIcon   = document.getElementById("verdict-icon");
const verdictLabel  = document.getElementById("verdict-label");
const meterArc      = document.getElementById("meter-arc");
const confidencePct = document.getElementById("confidence-pct");
const barFake       = document.getElementById("bar-fake");
const barGenuine    = document.getElementById("bar-genuine");
const pctFake       = document.getElementById("pct-fake");
const pctGenuine    = document.getElementById("pct-genuine");
const signalChips   = document.getElementById("signal-chips");
const errorMessage  = document.getElementById("error-message");

// Character counter
reviewInput.addEventListener("input", () => {
  const len = reviewInput.value.length;
  charCount.textContent = `${len.toLocaleString()} / 10,000`;
  charCount.classList.remove("warn", "danger");
  if (len >= MAX_CHARS)       charCount.classList.add("danger");
  else if (len >= WARN_CHARS) charCount.classList.add("warn");
});

// Ctrl+Enter shortcut
reviewInput.addEventListener("keydown", (e) => {
  if (e.ctrlKey && e.key === "Enter") { e.preventDefault(); runPrediction(); }
});

btnClear.addEventListener("click", () => {
  reviewInput.value = "";
  charCount.textContent = "0 / 10,000";
  charCount.classList.remove("warn", "danger");
  showSection(stateIdle);
  reviewInput.focus();
});

btnAnalyse.addEventListener("click", runPrediction);

async function runPrediction() {
  const text = reviewInput.value.trim();
  if (!text) { showError("Please enter a review before analysing."); return; }
  if (text.length > MAX_CHARS) { showError(`Review exceeds ${MAX_CHARS.toLocaleString()} characters.`); return; }

  setSingleLoading(true);
  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ review_text: text }),
    });
    const data = await resp.json();
    if (!resp.ok) { showError(data.error || `Server error (HTTP ${resp.status}).`); return; }
    if (!data.prediction || data.prediction === "Unknown") { showError("Unexpected server response."); return; }
    renderResult(data, text);
  } catch (err) {
    showError("Network error — could not reach the server.");
  } finally {
    setSingleLoading(false);
  }
}

function renderResult(data, rawText) {
  const isFake = data.prediction === "Fake";
  verdictIcon.textContent  = isFake ? "🚨" : "✅";
  verdictLabel.textContent = data.prediction;
  verdictLabel.className   = "verdict-label " + (isFake ? "fake" : "genuine");

  const cs = getComputedStyle(document.documentElement);
  meterArc.style.stroke = isFake
    ? cs.getPropertyValue("--fake").trim()
    : cs.getPropertyValue("--genuine").trim();
  confidencePct.textContent = `${data.confidence_pct}%`;
  meterArc.style.strokeDashoffset = CIRCUMFERENCE;
  setTimeout(() => {
    meterArc.style.strokeDashoffset = CIRCUMFERENCE * (1 - data.confidence);
  }, 50);

  barFake.style.width    = `${data.fake_prob}%`;
  barGenuine.style.width = `${data.genuine_prob}%`;
  pctFake.textContent    = `${data.fake_prob}%`;
  pctGenuine.textContent = `${data.genuine_prob}%`;

  renderSignalChips(computeSignals(rawText));
  showSection(stateResult);
}

function renderSignalChips(signals) {
  signalChips.innerHTML = "";
  if (!signals.length) {
    const chip = document.createElement("span");
    chip.className = "chip chip-ok";
    chip.textContent = "✓ No obvious red-flag patterns detected";
    signalChips.appendChild(chip);
    return;
  }
  signals.forEach((msg) => {
    const chip = document.createElement("span");
    chip.className = "chip chip-flag";
    chip.textContent = "⚑ " + msg;
    signalChips.appendChild(chip);
  });
}

function computeSignals(text) {
  const flags = [];
  const exclamations = (text.match(/!/g) || []).length;
  if (exclamations >= 3) flags.push(`Excessive exclamation marks (${exclamations} found)`);

  const capsWords = (text.match(/\b[A-Z]{3,}\b/g) || []).length;
  if (capsWords >= 3) flags.push(`Multiple ALL-CAPS words (${capsWords} found)`);

  const superlatives = [
    "best ever","changed my life","absolutely perfect","best purchase","best product",
    "totally worth","life changing","life-changing","never been happier",
    "best i have ever","best i've ever","most amazing","hands down the best",
    "cannot believe how","absolutely incredible","absolutely amazing",
    "best on the market","worth every penny","totally perfect","completely transformed",
  ];
  const lower = text.toLowerCase();
  const matched = superlatives.filter((p) => lower.includes(p));
  if (matched.length >= 2) flags.push(`Superlative overload: "${matched.slice(0,2).join('", "')}"`);

  const wordCount = text.trim().split(/\s+/).filter(Boolean).length;
  if (wordCount < 15) flags.push(`Very short review (${wordCount} words)`);

  const qMarks = (text.match(/\?/g) || []).length;
  if (wordCount > 40 && qMarks === 0) flags.push("Long review with no questions (one-sided tone)");

  return flags;
}

function showSection(activeEl) {
  [stateIdle, stateResult, stateError].forEach((el) => {
    el === activeEl ? el.classList.remove("hidden") : el.classList.add("hidden");
  });
}
function showError(msg) { errorMessage.textContent = msg; showSection(stateError); }
function setSingleLoading(on) {
  btnAnalyse.disabled = on; btnClear.disabled = on; reviewInput.disabled = on;
  on ? btnAnalyse.classList.add("loading") : btnAnalyse.classList.remove("loading");
}

// ── Init ──────────────────────────────────────────────────────────
showSection(stateIdle);
showUrlState(urlIdle);
