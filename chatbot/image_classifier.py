"""
=============================================================
  AgreeCulture - Image Recognition Module
=============================================================
  Strategy (most accurate → fallback):
  1. Gemini Vision  — REST API with base64 image (multimodal)
     Tries 3 Gemini models; rotates on 429 quota errors.
  2. OpenRouter Vision — tries models that support vision.
  3. Color-histogram KNN — offline fallback (no API needed).

  Response format is identical across all three so Flask
  route and frontend JS need zero changes.
=============================================================
"""

import base64
import io
import json
import os
import re
import time

import numpy as np
import requests
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge base (loaded once at import)
# ─────────────────────────────────────────────────────────────────────────────

def _load_kb() -> dict:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    kb_path  = os.path.join(base_dir, "data", "knowledge_base.json")
    try:
        with open(kb_path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


_KB = _load_kb()
_KNOWN_CROPS = list(_KB.get("crops", {}).keys())

_CROP_LABELS = {
    "rice":      "Rice Plant",
    "wheat":     "Wheat Plant",
    "maize":     "Maize (Corn) Plant",
    "tomato":    "Tomato Plant",
    "potato":    "Potato Plant",
    "cotton":    "Cotton Plant",
    "sugarcane": "Sugarcane Plant",
    "mango":     "Mango Tree",
    "groundnut": "Groundnut (Peanut) Plant",
    "soybean":   "Soybean Plant",
}


def _get_crop_info(crop_key: str) -> dict:
    info = _KB.get("crops", {}).get(crop_key, {})
    return {
        "season":            info.get("season",            "N/A"),
        "soil_type":         info.get("soil_type",         "N/A"),
        "water_requirement": info.get("water_requirement", "N/A"),
        "fertilizer":        info.get("fertilizer",        "N/A"),
        "yield":             info.get("yield",             "N/A"),
        "tips":              info.get("tips",              "N/A"),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Shared vision prompt
# ─────────────────────────────────────────────────────────────────────────────

_VISION_PROMPT = f"""You are an expert botanist and agricultural scientist.
Analyze this image and identify the plant shown.

Known crops in our database: {', '.join(_KNOWN_CROPS)}

Return ONLY a JSON object (no markdown fences, no extra text):

{{
  "crop_key": "potato",
  "plant_label": "Potato Plant",
  "confidence": 91,
  "top3": [
    {{"crop": "potato", "confidence": 91}},
    {{"crop": "tomato", "confidence": 6}},
    {{"crop": "groundnut", "confidence": 3}}
  ],
  "is_plant": true,
  "note": "Potato plant with tubers visible at the base"
}}

Rules:
- crop_key must be a single lowercase word matching one of the known crops if possible.
- confidence is an integer 0-100.
- top3 lists your 3 best guesses with confidence scores.
- is_plant is false ONLY if there is clearly no plant in the image at all.
- note is one short sentence describing what you see.
- If unsure, still give your best guess with lower confidence rather than saying unknown.
"""


def _parse_vision_json(raw: str) -> dict | None:
    """Strip markdown fences and parse JSON from model response."""
    text = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        parsed = json.loads(text)
        if "crop_key" in parsed and "confidence" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


def _image_to_b64(image_path: str) -> tuple[str, str]:
    """Return (mime_type, base64_string) for the image. Resizes to ≤1024px to save tokens."""
    img = Image.open(image_path).convert("RGB")
    # Downscale if too large (saves API tokens / bytes)
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return "image/jpeg", b64


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY: Gemini Vision via REST API (no SDK quota tracking issues)
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_REST_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
    "/{model}:generateContent?key={key}"
)

_GEMINI_VISION_MODELS = [
    "gemini-2.0-flash",           # primary — best vision support
    "gemini-2.0-flash-lite",      # lighter model, separate quota pool
    "gemini-2.5-flash",           # Gemini 2.5 Flash (latest stable)
]


def _predict_via_gemini(image_path: str) -> dict | None:
    """
    Call Gemini Vision via the REST API (not the Python SDK).
    Tries multiple models. Returns parsed dict or None.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    mime, b64_data = _image_to_b64(image_path)

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": b64_data,
                        }
                    },
                    {
                        "text": _VISION_PROMPT
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 512,
        }
    }

    for model in _GEMINI_VISION_MODELS:
        url = _GEMINI_REST_URL.format(model=model, key=api_key)
        try:
            resp = requests.post(url, json=payload, timeout=30)

            if resp.status_code == 429:
                print(f"[WARN] Gemini Vision {model} quota exhausted, trying next...")
                continue

            if resp.status_code != 200:
                print(f"[WARN] Gemini Vision {model} error {resp.status_code}: {resp.text[:150]}")
                continue

            data = resp.json()
            raw  = (
                data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
            )

            parsed = _parse_vision_json(raw)
            if parsed:
                parsed["_powered_by"] = "gemini-vision"
                print(f"[OK] Gemini Vision REST ({model}): {parsed['crop_key']} @ {parsed['confidence']}%")
                return parsed
            else:
                print(f"[WARN] Gemini Vision {model} returned unparseable JSON: {raw[:150]}")
                continue

        except requests.exceptions.Timeout:
            print(f"[WARN] Gemini Vision {model} timed out.")
            continue
        except Exception as e:
            print(f"[ERROR] Gemini Vision {model}: {e}")
            continue

    print("[WARN] All Gemini Vision models failed or quota exhausted.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  SECONDARY: OpenRouter Vision (free models that support images)
# ─────────────────────────────────────────────────────────────────────────────

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Free OpenRouter models that support vision/image input
# NOTE: Most "free" vision models on OpenRouter are deprecated or offline.
# nvidia/nemotron-nano-12b-v2-vl:free is currently the only confirmed live free vision model.
_OPENROUTER_VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",  # NVIDIA VL model — confirmed live (May 2026)
    "google/gemma-4-26b-a4b-it:free",       # Gemma 4 (vision capable, may be rate-limited)
    "google/gemma-4-31b-it:free",           # Gemma 4 large (vision capable, fallback)
]


def _predict_via_openrouter(image_path: str) -> dict | None:
    """
    Call OpenRouter free vision models with a base64-encoded image.
    Returns parsed dict or None.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key or api_key == "your_openrouter_api_key_here":
        return None

    mime, b64_data = _image_to_b64(image_path)
    data_url = f"data:{mime};base64,{b64_data}"

    headers = {
        "Authorization":  f"Bearer {api_key}",
        "Content-Type":   "application/json",
        "HTTP-Referer":   "https://agreeculture.app",
        "X-Title":        "AgreeCulture Plant ID",
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {
                    "type": "text",
                    "text": _VISION_PROMPT,
                },
            ],
        }
    ]

    for model in _OPENROUTER_VISION_MODELS:
        try:
            payload = {
                "model":       model,
                "messages":    messages,
                "temperature": 0.2,
                "max_tokens":  512,
            }
            resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                print(f"[WARN] OpenRouter vision {model} rate-limited, waiting 4s and retrying...")
                time.sleep(4)
                # Retry once after waiting
                resp = requests.post(_OPENROUTER_URL, headers=headers, json=payload, timeout=30)
                if resp.status_code != 200:
                    print(f"[WARN] OpenRouter vision {model} still failing ({resp.status_code}), trying next...")
                    continue

            if resp.status_code != 200:
                print(f"[WARN] OpenRouter vision {model} error {resp.status_code}: {resp.text[:150]}")
                continue

            raw = resp.json()["choices"][0]["message"].get("content")
            if not raw:
                print(f"[WARN] OpenRouter vision {model} returned empty/null content, trying next...")
                continue
            raw = raw.strip()
            parsed = _parse_vision_json(raw)
            if parsed:
                parsed["_powered_by"] = "openrouter-vision"
                print(f"[OK] OpenRouter Vision ({model}): {parsed['crop_key']} @ {parsed['confidence']}%")
                return parsed
            else:
                print(f"[WARN] OpenRouter vision {model} returned unparseable: {raw[:150]}")
                continue

        except requests.exceptions.Timeout:
            print(f"[WARN] OpenRouter vision {model} timed out.")
            continue
        except Exception as e:
            print(f"[ERROR] OpenRouter vision {model}: {e}")
            continue

    print("[WARN] All OpenRouter vision models failed.")
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK: Color-histogram KNN (fully offline, no API needed)
# ─────────────────────────────────────────────────────────────────────────────

_PLANT_COLOR_PROFILES = {
    "rice":      {"R": (100, 160), "G": (120, 180), "B": (60, 110)},
    "wheat":     {"R": (180, 220), "G": (160, 200), "B": (80, 130)},
    "maize":     {"R": (90,  140), "G": (140, 200), "B": (50,  90)},
    "tomato":    {"R": (180, 255), "G": (40,  100), "B": (30,  80)},
    "potato":    {"R": (130, 190), "G": (120, 180), "B": (60, 120)},
    "cotton":    {"R": (220, 255), "G": (220, 255), "B": (210, 255)},
    "sugarcane": {"R": (70,  120), "G": (130, 190), "B": (50,  90)},
    "mango":     {"R": (60,  110), "G": (120, 180), "B": (40,  90)},
    "groundnut": {"R": (110, 160), "G": (130, 190), "B": (60, 110)},
    "soybean":   {"R": (100, 150), "G": (140, 200), "B": (55, 100)},
}


def _build_color_knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X, y = [], []
    for crop_key, profile in _PLANT_COLOR_PROFILES.items():
        for _ in range(150):
            r = np.random.uniform(profile["R"][0], profile["R"][1])
            g = np.random.uniform(profile["G"][0], profile["G"][1])
            b = np.random.uniform(profile["B"][0], profile["B"][1])
            r_std = np.random.uniform(20, 50)
            g_std = np.random.uniform(20, 50)
            b_std = np.random.uniform(15, 40)
            green_dom = g / (r + b + 1e-5)
            X.append([r, g, b, r_std, g_std, b_std, green_dom])
            y.append(crop_key)

    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=7, weights="distance")
    clf.fit(X_scaled, y)
    return clf, scaler


_color_clf, _color_scaler = _build_color_knn()


def _extract_color_features(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    arr = np.array(img, dtype=np.float32)
    R, G, B = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    return np.array([
        R.mean(), G.mean(), B.mean(),
        R.std(),  G.std(),  B.std(),
        G.mean() / (R.mean() + B.mean() + 1e-5),
    ], dtype=np.float32)


def _predict_via_color_knn(image_path: str) -> dict:
    feats     = _extract_color_features(image_path)
    scaled    = _color_scaler.transform([feats])
    crop_key  = _color_clf.predict(scaled)[0]
    proba_map = dict(zip(_color_clf.classes_, _color_clf.predict_proba(scaled)[0]))
    confidence = round(proba_map[crop_key] * 100, 1)
    top3_raw   = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)[:3]
    top3       = [{"crop": k, "confidence": round(v * 100, 1)} for k, v in top3_raw]
    return {
        "crop_key":    crop_key,
        "plant_label": _CROP_LABELS.get(crop_key, crop_key.title() + " Plant"),
        "confidence":  confidence,
        "top3":        top3,
        "is_plant":    True,
        "note":        "[Offline] All AI vision APIs are currently unavailable. Using basic color analysis - accuracy is limited.",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — called by Flask /identify-plant route
# ─────────────────────────────────────────────────────────────────────────────

def predict_plant(image_path: str) -> dict:
    """
    Main entry point.
    Tries: Gemini Vision REST → OpenRouter Vision → color-KNN fallback.
    Always returns a dict the frontend can render.
    """
    try:
        # ── 1. Gemini Vision (REST API) ───────────────────────────────────
        result = _predict_via_gemini(image_path)

        # ── 2. OpenRouter Vision ──────────────────────────────────────────
        if result is None:
            result = _predict_via_openrouter(image_path)

        # ── 3. Color-histogram KNN fallback ──────────────────────────────
        if result is None:
            print("[INFO] All vision APIs unavailable — using color-KNN fallback.")
            result = _predict_via_color_knn(image_path)
            result["_powered_by"] = "color-knn-fallback"

        # Extract the engine tag injected by each sub-function
        powered_by  = result.pop("_powered_by", "ai-vision")
        crop_key    = result.get("crop_key", "unknown").lower().strip()
        plant_label = result.get(
            "plant_label",
            _CROP_LABELS.get(crop_key, crop_key.replace("_", " ").title() + " Plant")
        )
        confidence = int(result.get("confidence", 50))
        is_plant   = result.get("is_plant", True)
        note       = result.get("note", "")

        # Normalize top3
        top3 = []
        for item in result.get("top3", [])[:3]:
            c = item.get("crop") or item.get("crop_key") or crop_key
            v = item.get("confidence", 0)
            top3.append({"crop": c, "confidence": round(float(v), 1)})

        if not top3:
            top3 = [{"crop": crop_key, "confidence": confidence}]

        if not is_plant:
            return {
                "success": False,
                "error":   "No plant detected",
                "message": "The image doesn't appear to contain a plant. "
                           "Please upload a clear photo of a crop or plant. 🌱",
            }

        return {
            "success":        True,
            "powered_by":     powered_by,
            "predicted_crop": crop_key,
            "plant_label":    plant_label,
            "confidence":     confidence,
            "top3":           top3,
            "note":           note,
            "crop_info":      _get_crop_info(crop_key),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error":   str(e),
            "message": "Could not process the image. Please upload a clear plant photo.",
        }
