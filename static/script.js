// script.js
"use strict";

const CIRCUMFERENCE = 339.29; // 2 * Math.PI * 54
const MAX_CHARS     = 10000;
const WARN_CHARS    = 9000;

// ── Settings panel ────────────────────────────────────────────────
const btnSettings      = document.getElementById("btn-settings");
const settingsOverlay  = document.getElementById("settings-overlay");
const btnCloseSettings = document.getElementById("btn-close-settings");
const geminiKeyInput   = document.getElementById("gemini-key-input");
const btnSaveKey       = document.getElementById("btn-save-key");
const btnClearKey      = document.getElementById("btn-clear-key");
const aiStatusBadge    = document.getElementById("ai-status-badge");
const playwrightBadge  = document.getElementById("playwright-badge");
const stGeminiLib      = document.getElementById("st-gemini-lib");
const stGeminiKey      = document.getElementById("st-gemini-key");
const stPlaywright     = document.getElementById("st-playwright");

btnSettings.addEventListener("click", () => {
  settingsOverlay.classList.remove("hidden");
  refreshGeminiStatus();
});
btnCloseSettings.addEventListener("click", () => settingsOverlay.classList.add("hidden"));
settingsOverlay.addEventListener("click", (e) => {
  if (e.target === settingsOverlay) settingsOverlay.classList.add("hidden");
});

async function refreshGeminiStatus() {
  try {
    const r = await fetch("/gemini-status");
    const d = await r.json();
    // Dots
    stGeminiLib.className  = "status-dot " + (d.gemini_library ? "dot-green" : "dot-red");
    stGeminiKey.className  = "status-dot " + (d.gemini_active  ? "dot-green" : "dot-grey");
    stPlaywright.className = "status-dot " + (d.playwright_library ? "dot-green" : "dot-grey");
    // Header badges
    if (d.gemini_active) {
      aiStatusBadge.textContent = "🤖 AI: On";
      aiStatusBadge.classList.add("active");
    } else {
      aiStatusBadge.textContent = "🤖 AI: Off";
      aiStatusBadge.classList.remove("active");
    }
    if (d.playwright_library) {
      playwrightBadge.classList.remove("hidden");
    } else {
      playwrightBadge.classList.add("hidden");
    }
  } catch (_) {}
}

btnSaveKey.addEventListener("click", async () => {
  const key = geminiKeyInput.value.trim();
  if (!key) { alert("Please enter an API key."); return; }
  btnSaveKey.disabled = true;
  btnSaveKey.textContent = "Saving…";
  try {
    const r = await fetch("/set-api-key", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ api_key: key }),
    });
    const d = await r.json();
    refreshGeminiStatus();
    if (d.gemini_active) {
      settingsOverlay.classList.add("hidden");
    }
  } catch (_) {
    alert("Could not reach the server.");
  } finally {
    btnSaveKey.disabled = false;
    btnSaveKey.textContent = "Save";
  }
});

btnClearKey.addEventListener("click", async () => {
  geminiKeyInput.value = "";
  await fetch("/set-api-key", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: "" }),
  });
  refreshGeminiStatus();
});

// Poll status on page load
refreshGeminiStatus();

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
  // Trim whitespace and strip any accidental surrounding quotes
  let url = urlInput.value.trim().replace(/^["']|["']$/g, "");

  if (!url) { showUrlError("Please enter a product URL."); return; }

  // Auto-prepend https:// if user forgot the scheme
  if (!/^https?:\/\//i.test(url)) {
    url = "https://" + url;
    urlInput.value = url;
  }

  // Basic domain check — must have at least one dot after the scheme
  if (!/^https?:\/\/[^/\s]+\.[^/\s]/.test(url)) {
    showUrlError("That doesn't look like a valid URL. Please paste the full product page link.");
    return;
  }

  setUrlLoading(true);
  showUrlState(urlLoading);

  try {
    const resp = await fetch("/analyse-url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });

    let data;
    try {
      data = await resp.json();
    } catch (_) {
      showUrlError(`Server returned an unreadable response (HTTP ${resp.status}). Try a different URL.`);
      return;
    }

    if (!resp.ok) {
      showUrlError(data.error || `Server error (HTTP ${resp.status}).`);
      return;
    }
    // Handle "soft errors" — server returned 200 but with an error field (e.g. no reviews found)
    if (data.error) {
      showUrlError(data.error);
      return;
    }
    renderUrlResult(data);
  } catch (err) {
    if (err.name === "TypeError") {
      showUrlError("Network error — make sure the Flask server is running on port 5000.");
    } else {
      showUrlError("Unexpected error: " + err.message);
    }
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

  // Verdict chip with trust score
  urlVerdictChip.className = "url-verdict-chip";
  const trustScore = data.aggregate_trust || 50;
  if (data.genuine_pct >= 70 && trustScore >= 60) {
    urlVerdictChip.classList.add("mostly-genuine");
    urlVerdictChip.textContent = `✅ Mostly Genuine (Trust: ${trustScore}%)`;
  } else if (data.fake_pct >= 70 || trustScore < 30) {
    urlVerdictChip.classList.add("mostly-fake");
    urlVerdictChip.textContent = `🚨 Mostly Fake (Trust: ${trustScore}%)`;
  } else {
    urlVerdictChip.classList.add("mixed");
    urlVerdictChip.textContent = `⚖️ Mixed Reviews (Trust: ${trustScore}%)`;
  }

  // Update header AI badge from actual response
  if (data.gemini_active) {
    aiStatusBadge.textContent = "🤖 AI: On";
    aiStatusBadge.classList.add("active");
  }
  if (data.playwright_active) {
    playwrightBadge.classList.remove("hidden");
  }

  // Display suspicious patterns if any
  renderSuspiciousPatterns(data.suspicious_patterns || []);

  // Render review list
  renderReviewList(allReviews);
  showUrlState(urlResultBody);
}

function renderSuspiciousPatterns(patterns) {
  const container = document.getElementById("suspicious-patterns-container");
  if (!container) return;
  
  if (!patterns || patterns.length === 0) {
    container.innerHTML = "";
    container.style.display = "none";
    return;
  }
  
  container.style.display = "block";
  container.innerHTML = `
    <div class="suspicious-patterns-box">
      <div class="patterns-header">
        <span class="patterns-icon">🔍</span>
        <span class="patterns-title">Intelligence Analysis</span>
      </div>
      <div class="patterns-list">
        ${patterns.map(p => `<div class="pattern-item">${escapeHtml(p)}</div>`).join("")}
      </div>
    </div>
  `;
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

    const trustScore = r.trust_score || 50;
    const trustClass = trustScore >= 70 ? "high-trust" : trustScore >= 40 ? "medium-trust" : "low-trust";
    const verificationBadge = r.verification_status
      ? `<span class="verification-badge ${trustClass}" title="${r.verification_status}">Trust: ${trustScore}%</span>`
      : "";

    // AI verdict chip
    let aiChipHtml = "";
    if (r.ai_verdict) {
      const av = r.ai_verdict;
      const isConflict = r.verification_status === "ai_conflict";
      let chipClass = isConflict ? "ai-conflict" : (av.authentic ? "ai-genuine" : "ai-fake");
      let chipLabel = isConflict
        ? `⚡ AI Conflict (${av.confidence}%)`
        : (av.authentic ? `🤖 AI: Authentic (${av.confidence}%)` : `🤖 AI: Suspicious (${av.confidence}%)`);
      aiChipHtml = `
        <span class="ai-chip ${chipClass}">${chipLabel}</span>
        ${av.reason ? `<span class="ai-reason">"${escapeHtml(av.reason)}"</span>` : ""}
      `;
    }

    const flagsHtml = r.flags && r.flags.length > 0
      ? `<div class="review-flags">${r.flags.slice(0, 3).map(f => `<span class="flag-chip">⚑ ${escapeHtml(f)}</span>`).join("")}</div>`
      : "";

    item.innerHTML = `
      <div class="review-text-col">
        <p class="review-text-body" id="rtb-${idx}">${escapeHtml(r.text)}</p>
        ${needsExpand ? `<button class="review-expand-btn" data-idx="${idx}" type="button">Show more</button>` : ""}
        ${flagsHtml}
        ${aiChipHtml}
        <span class="review-meta">${r.confidence_pct}% ML confidence · Fake ${r.fake_prob}% · Genuine ${r.genuine_prob}%</span>
      </div>
      <div class="review-badges">
        <span class="review-badge ${r.prediction === "Fake" ? "fake" : "genuine"}">
          ${r.prediction === "Fake" ? "🚨 Fake" : "✅ Genuine"}
        </span>
        ${verificationBadge}
      </div>`;
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
  // Render newlines as line breaks
  urlErrorMsg.innerHTML = msg.replace(/\n/g, "<br>").replace(/•/g, "&#8226;");
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

  // Display trust score and verification status if available
  const trustScoreEl = document.getElementById("trust-score-display");
  if (trustScoreEl && data.trust_score !== undefined) {
    const trustScore = data.trust_score;
    const trustClass = trustScore >= 70 ? "high-trust" : trustScore >= 40 ? "medium-trust" : "low-trust";
    trustScoreEl.innerHTML = `
      <div class="trust-score-box ${trustClass}">
        <span class="trust-label">Trust Score:</span>
        <span class="trust-value">${trustScore}%</span>
      </div>
    `;
    if (data.confidence_note) {
      trustScoreEl.innerHTML += `<p class="confidence-note">${escapeHtml(data.confidence_note)}</p>`;
    }
    // Append Gemini AI verdict chip if present
    if (data.ai_verdict) {
      const av = data.ai_verdict;
      const isConflict = data.verification_status === "ai_conflict";
      const chipClass = isConflict ? "ai-conflict" : (av.authentic ? "ai-genuine" : "ai-fake");
      const chipLabel = isConflict
        ? `⚡ AI Conflict (${av.confidence}%)`
        : (av.authentic ? `🤖 AI: Authentic (${av.confidence}%)` : `🤖 AI: Suspicious (${av.confidence}%)`);
      trustScoreEl.innerHTML += `
        <div style="margin-top:10px;display:flex;flex-direction:column;gap:4px;align-items:flex-start;">
          <span class="ai-chip ${chipClass}">${chipLabel}</span>
          ${av.reason ? `<span class="ai-reason">"${escapeHtml(av.reason)}"</span>` : ""}
        </div>`;
    }
  }

  // Update AI badge if Gemini was used
  if (data.gemini_active) {
    aiStatusBadge.textContent = "🤖 AI: On";
    aiStatusBadge.classList.add("active");
  }

  renderSignalChips(data.flags || computeSignals(rawText));
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
