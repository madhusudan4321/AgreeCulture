/* =====================================================
   AgreeCulture — Frontend JavaScript
   Handles: Crop Prediction, Chatbot, Image Upload, Stats
   ===================================================== */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
//  Mobile Navigation
// ─────────────────────────────────────────────────────────────────────────────

function toggleMobileNav() {
  const nav  = document.getElementById("nav-links");
  const btn  = document.getElementById("hamburger");
  nav.classList.toggle("open");
  btn.classList.toggle("open");
  btn.setAttribute("aria-expanded", nav.classList.contains("open"));
}

// Close mobile nav when any nav link is clicked
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", () => {
      document.getElementById("nav-links").classList.remove("open");
      document.getElementById("hamburger").classList.remove("open");
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
//  Crop emoji mapping for visual display
// ─────────────────────────────────────────────────────────────────────────────
const CROP_EMOJI = {
  rice: "🌾", wheat: "🌿", maize: "🌽", cotton: "☁️",
  sugarcane: "🎋", soybean: "🫘", potato: "🥔",
  tomato: "🍅", groundnut: "🥜", mango: "🥭"
};

function getCropEmoji(crop) {
  return CROP_EMOJI[crop?.toLowerCase()] || "🌱";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Preset values for quick testing of the crop form
// ─────────────────────────────────────────────────────────────────────────────
const PRESETS = {
  rice:  { nitrogen: 80, phosphorus: 45, potassium: 40, temperature: 25, humidity: 82, rainfall: 200 },
  wheat: { nitrogen: 100, phosphorus: 40, potassium: 42, temperature: 20, humidity: 65, rainfall: 80 },
  maize: { nitrogen: 85, phosphorus: 58, potassium: 43, temperature: 25, humidity: 65, rainfall: 100 },
};

function setPreset(cropName) {
  const p = PRESETS[cropName];
  if (!p) return;
  document.getElementById("nitrogen").value    = p.nitrogen;
  document.getElementById("phosphorus").value  = p.phosphorus;
  document.getElementById("potassium").value   = p.potassium;
  document.getElementById("temperature").value = p.temperature;
  document.getElementById("humidity").value    = p.humidity;
  document.getElementById("rainfall").value    = p.rainfall;
}

// ─────────────────────────────────────────────────────────────────────────────
//  CROP PREDICTION
// ─────────────────────────────────────────────────────────────────────────────

document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();

  const btn = document.getElementById("predict-btn");
  const btnText = document.getElementById("predict-btn-text");
  btnText.innerHTML = '<span class="spinner"></span> Analyzing...';
  btn.disabled = true;

  // Collect form data
  const formData = new FormData(this);

  try {
    const response = await fetch("/predict", { method: "POST", body: formData });
    const data = await response.json();
    displayPredictionResult(data);

    // If prediction succeeded, ask Gemini for a personalized insight
    if (data.status === "success" && data.final_recommendation) {
      fetchGeminiInsight(data.final_recommendation, Object.fromEntries(formData));
    }
  } catch (error) {
    alert("Error connecting to server. Make sure Flask is running!");
    console.error(error);
  } finally {
    btnText.textContent = "🚀 Predict Best Crop";
    btn.disabled = false;
  }
});

function displayPredictionResult(data) {
  // Hide placeholder, show result card
  document.getElementById("result-placeholder").style.display = "none";
  const resultCard = document.getElementById("predict-result");
  resultCard.style.display = "block";
  resultCard.style.animation = "none";
  setTimeout(() => { resultCard.style.animation = "fadeSlideIn 0.4s ease"; }, 10);

  if (data.status === "demo" || data.status === "success") {
    const crop = data.final_recommendation || data.demo_prediction;

    // Update best crop box
    document.getElementById("best-crop-emoji").textContent = getCropEmoji(crop);
    document.getElementById("best-crop-name").textContent  = crop.charAt(0).toUpperCase() + crop.slice(1);

    // Model votes
    const votesContainer = document.getElementById("model-votes-list");
    votesContainer.innerHTML = "";

    if (data.model_predictions) {
      for (const [modelName, pred] of Object.entries(data.model_predictions)) {
        const row = document.createElement("div");
        row.className = "vote-row";
        row.innerHTML = `
          <span class="vote-model">${modelName}</span>
          <span class="vote-crop">${getCropEmoji(pred.crop)} ${pred.crop}</span>
          ${pred.confidence ? `<span class="vote-conf">${pred.confidence}%</span>` : ""}
        `;
        votesContainer.appendChild(row);
      }
    } else {
      votesContainer.innerHTML = `<div style="color:#888;font-size:0.85rem;padding:0.5rem 0;">
        ⚠️ ${data.message || "Train models first to see predictions."}</div>`;
    }

    // Crop info
    const info = data.crop_info;
    if (info) {
      document.getElementById("crop-details").innerHTML = `
        <div class="detail-row"><span class="detail-icon">📅</span> <span class="detail-val"><b>Season:</b> ${info.season}</span></div>
        <div class="detail-row"><span class="detail-icon">🏔️</span> <span class="detail-val"><b>Soil:</b> ${info.soil_type}</span></div>
        <div class="detail-row"><span class="detail-icon">💧</span> <span class="detail-val"><b>Water:</b> ${info.water_requirement}</span></div>
        <div class="detail-row"><span class="detail-icon">🧪</span> <span class="detail-val"><b>Fertilizer:</b> ${info.fertilizer}</span></div>
        <div class="detail-row"><span class="detail-icon">📦</span> <span class="detail-val"><b>Yield:</b> ${info.yield}</span></div>
        <div class="detail-row"><span class="detail-icon">💡</span> <span class="detail-val">${info.tips}</span></div>
      `;
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  CHATBOT
// ─────────────────────────────────────────────────────────────────────────────

async function sendChat() {
  const input = document.getElementById("chat-input");
  const message = input.value.trim();
  if (!message) return;

  input.value = "";

  // Add user message to chat
  appendChatMessage("user", message);

  // Show typing indicator
  const typingId = showTypingIndicator();

  const sessionId = "main-" + (localStorage.getItem("agreeCultureSession") || (() => {
    const id = Date.now().toString(36);
    localStorage.setItem("agreeCultureSession", id);
    return id;
  })());

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: "user-session" })
    });
    const data = await response.json();

    removeTypingIndicator(typingId);
    appendChatMessage("bot", data.bot_response || "Sorry, I couldn't process that.", data.powered_by);
  } catch (error) {
    removeTypingIndicator(typingId);
    appendChatMessage("bot", "⚠️ Error connecting to server. Is Flask running?");
  }
}

function appendChatMessage(role, text, poweredBy) {
  const chatWindow = document.getElementById("chat-window");

  const messageDiv = document.createElement("div");
  messageDiv.className = `chat-message ${role}`;

  const now     = new Date();
  const timeStr = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // Convert newlines to <br> and handle bold markdown (**)
  const formattedText = text
    .replace(/\n/g, "<br/>")
    .replace(/\*\*(.*?)\*\*/g, "<b>$1</b>");

  // Show which engine responded (only for bot)
  let engineTag = "";
  if (role === "bot" && poweredBy === "gemini") {
    engineTag = `<span class="chat-engine-tag">✨ Gemini AI</span>`;
  } else if (role === "bot" && poweredBy === "openrouter") {
    engineTag = `<span class="chat-engine-tag openrouter">&#9733; OpenRouter</span>`;
  } else if (role === "bot" && poweredBy === "fallback") {
    engineTag = `<span class="chat-engine-tag fallback">&#9881; Local</span>`;
  }

  messageDiv.innerHTML = `
    <div class="chat-bubble">${formattedText}${engineTag}</div>
    <div class="chat-time">${role === "bot" ? "Assistant" : "You"} • ${timeStr}</div>
  `;

  chatWindow.appendChild(messageDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showTypingIndicator() {
  const chatWindow = document.getElementById("chat-window");
  const id = "typing-" + Date.now();

  const typingDiv = document.createElement("div");
  typingDiv.className = "chat-message bot";
  typingDiv.id = id;
  typingDiv.innerHTML = `
    <div class="chat-bubble">
      <div class="typing-indicator">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  chatWindow.appendChild(typingDiv);
  chatWindow.scrollTop = chatWindow.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function sendSuggestion(text) {
  document.getElementById("chat-input").value = text;
  sendChat();
}


// ─────────────────────────────────────────────────────────────────────────────
//  ML STATS — Load on page load
// ─────────────────────────────────────────────────────────────────────────────

async function loadModelStats() {
  try {
    const response = await fetch("/model-stats");
    const data = await response.json();
    displayModelStats(data);
  } catch (e) {
    document.getElementById("stats-grid").innerHTML =
      "<div class='loading-card'>Could not load model stats.</div>";
  }
}

function displayModelStats(data) {
  const grid = document.getElementById("stats-grid");
  grid.innerHTML = "";

  // Find best accuracy
  const entries = Object.entries(data).filter(([k]) => k !== "note");
  const maxAcc = Math.max(...entries.map(([, v]) => v.accuracy));

  const modelIcons = {
    "Random Forest": "🌳",
    "Decision Tree": "🌲",
    "KNN": "📍"
  };

  const modelDesc = {
    "Random Forest": "Ensemble of 100 decision trees. Robust and accurate.",
    "Decision Tree": "Single tree with depth limit. Fast and interpretable.",
    "KNN": "7-nearest neighbors with distance weighting."
  };

  for (const [modelName, info] of entries) {
    const isBest = info.accuracy === maxAcc;
    const card = document.createElement("div");
    card.className = "stat-card" + (isBest ? " best-model" : "");
    card.innerHTML = `
      <div class="stat-model-name">${modelIcons[modelName] || "🤖"} ${modelName}</div>
      <div class="stat-accuracy">${info.accuracy}%</div>
      <div class="stat-label">Test Accuracy</div>
      <div style="font-size:0.8rem;color:#888;margin-top:0.5rem;">${modelDesc[modelName] || ""}</div>
      ${isBest ? '<div class="stat-badge">🏆 Best Model</div>' : ""}
    `;
    grid.appendChild(card);
  }

  if (data.note) {
    const note = document.createElement("div");
    note.style.cssText = "text-align:center;color:#aaa;font-size:0.8rem;grid-column:1/-1;margin-top:0.5rem;";
    note.textContent = "ℹ️ " + data.note;
    grid.appendChild(note);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Scroll animation (Intersection Observer)
// ─────────────────────────────────────────────────────────────────────────────

function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll(".card, .section-header, .stat-card").forEach((el) => {
    el.classList.add("fade-in");
    observer.observe(el);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
//  Active nav link on scroll
// ─────────────────────────────────────────────────────────────────────────────

function initNavHighlight() {
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-link");

  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        navLinks.forEach((link) => link.classList.remove("active"));
        const activeLink = document.querySelector(`.nav-link[href="#${entry.target.id}"]`);
        if (activeLink) activeLink.classList.add("active");
      }
    });
  }, { threshold: 0.4 });

  sections.forEach((s) => observer.observe(s));
}

// ─────────────────────────────────────────────────────────────────────────────
//  GEMINI AI — Status check & crop insight
// ─────────────────────────────────────────────────────────────────────────────

async function checkGeminiStatus() {
  try {
    const res  = await fetch("/chat-status");
    const data = await res.json();

    const badgeGemini    = document.getElementById("ai-badge");
    const badgeOpenRouter = document.getElementById("ai-badge-openrouter");
    const badgeFallback  = document.getElementById("ai-badge-fallback");

    // Reset all
    badgeGemini.style.display     = "none";
    badgeOpenRouter.style.display = "none";
    badgeFallback.style.display   = "none";

    if (data.available) {
      badgeGemini.style.display = "inline-flex";
    } else if (data.openrouter_available) {
      badgeOpenRouter.style.display = "inline-flex";
    } else {
      badgeFallback.style.display = "inline-flex";
    }
  } catch (_) { /* silently ignore */ }
}

async function fetchGeminiInsight(cropName, formValues) {
  const box     = document.getElementById("gemini-insight-box");
  const text    = document.getElementById("gemini-insight-text");
  const loading = document.getElementById("gemini-loading");

  // Show the panel in loading state
  box.style.display = "block";
  text.textContent  = "";
  loading.style.display = "inline";

  try {
    const res  = await fetch("/predict-insight", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        crop:        cropName,
        nitrogen:    formValues.nitrogen,
        phosphorus:  formValues.phosphorus,
        potassium:   formValues.potassium,
        temperature: formValues.temperature,
        humidity:    formValues.humidity,
        rainfall:    formValues.rainfall,
      })
    });
    const data = await res.json();

    loading.style.display = "none";

    if (data.status === "success" && data.insight) {
      text.textContent = data.insight;
    } else {
      box.style.display = "none";   // hide if Gemini not configured
    }
  } catch (_) {
    box.style.display = "none";
  }
}

// ─────────────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  loadModelStats();
  initScrollAnimations();
  initNavHighlight();
  checkGeminiStatus();
});
