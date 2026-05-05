"""
=============================================================
  AgreeCulture - Image Recognition Module
=============================================================
  Strategy (most accurate → fallback):
  1. Gemini Vision API  — sends the image to Gemini for real
     plant identification. Works on any real photograph.
  2. Color-histogram KNN — a very rough offline fallback used
     only when Gemini is unavailable.

  The response dict format is identical in both cases so the
  Flask route and frontend JS need zero changes.
=============================================================
"""

import base64
import json
import os
import re

import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  Knowledge base (loaded once)
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

# Canonical crop keys supported by the knowledge base
_KNOWN_CROPS = list(_KB.get("crops", {}).keys())

# Human-readable labels for each crop key
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
    """Return knowledge-base info dict for the given crop key."""
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
#  Helper: encode image for Gemini Vision
# ─────────────────────────────────────────────────────────────────────────────

def _image_to_base64(image_path: str) -> tuple[str, str]:
    """Return (mime_type, base64_data) for the given image file."""
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime_map = {
        "jpg":  "image/jpeg",
        "jpeg": "image/jpeg",
        "png":  "image/png",
        "gif":  "image/gif",
        "webp": "image/webp",
    }
    mime = mime_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return mime, data


# ─────────────────────────────────────────────────────────────────────────────
#  PRIMARY: Gemini Vision plant identification
# ─────────────────────────────────────────────────────────────────────────────

_GEMINI_VISION_PROMPT = f"""You are an expert botanist and agricultural scientist.
Analyze this image and identify the plant shown.

Known crops in our database: {', '.join(_KNOWN_CROPS)}

Instructions:
1. If the image shows a plant from the known crops list, use that exact key name.
2. If the plant is NOT in the known crops list, still identify it (use a short lowercase name).
3. Return ONLY a JSON object in this exact format — no markdown, no extra text:

{{
  "crop_key": "rice",
  "plant_label": "Rice Plant",
  "confidence": 88,
  "top3": [
    {{"crop": "rice", "confidence": 88}},
    {{"crop": "wheat", "confidence": 7}},
    {{"crop": "maize", "confidence": 5}}
  ],
  "is_plant": true,
  "note": "Healthy rice paddy in early vegetative stage"
}}

Rules:
- "crop_key" must be a lowercase word (no spaces).
- "confidence" is an integer 0-100 reflecting your certainty.
- "top3" lists the 3 most likely matches with confidence scores summing to ~100.
- "is_plant" is false if the image does not contain a plant at all.
- "note" is one short sentence describing what you see.
- If you are NOT certain, still give your best guess but lower the confidence.
"""


def _predict_via_gemini(image_path: str) -> dict | None:
    """
    Send the image to Gemini Vision and return a structured prediction dict.
    Returns None if Gemini is unavailable or returns an unparseable response.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        mime, b64_data = _image_to_base64(image_path)

        # Gemini models that support vision (try newest first)
        vision_models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash",
        ]

        for model_name in vision_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=base64.b64decode(b64_data),
                            mime_type=mime,
                        ),
                        _GEMINI_VISION_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=512,
                    ),
                )

                raw = response.text.strip()

                # Strip markdown fences if present
                raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE)
                raw = re.sub(r"\n?```$", "", raw)

                parsed = json.loads(raw)

                # Validate required fields
                if "crop_key" not in parsed or "confidence" not in parsed:
                    continue

                print(f"[OK] Gemini Vision ({model_name}): {parsed['crop_key']} @ {parsed['confidence']}%")
                return parsed

            except json.JSONDecodeError as je:
                print(f"[WARN] Gemini Vision JSON parse error ({model_name}): {je} | raw={raw[:200]}")
                continue
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    print(f"[WARN] Gemini Vision quota hit ({model_name}), trying next model...")
                    continue
                print(f"[WARN] Gemini Vision error ({model_name}): {err[:150]}")
                continue

    except ImportError:
        print("[WARN] google-genai not installed. Falling back to color classifier.")
    except Exception as e:
        print(f"[WARN] Gemini Vision init failed: {e}")

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  FALLBACK: Color-histogram KNN classifier (offline, rough)
# ─────────────────────────────────────────────────────────────────────────────

# Color profiles per crop (mean R, G, B ranges)
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
    feats  = _extract_color_features(image_path)
    scaled = _color_scaler.transform([feats])
    crop_key     = _color_clf.predict(scaled)[0]
    proba_map    = dict(zip(_color_clf.classes_, _color_clf.predict_proba(scaled)[0]))
    confidence   = round(proba_map[crop_key] * 100, 1)
    top3_raw     = sorted(proba_map.items(), key=lambda x: x[1], reverse=True)[:3]
    top3         = [{"crop": k, "confidence": round(v * 100, 1)} for k, v in top3_raw]
    return {
        "crop_key":    crop_key,
        "plant_label": _CROP_LABELS.get(crop_key, crop_key.title() + " Plant"),
        "confidence":  confidence,
        "top3":        top3,
        "is_plant":    True,
        "note":        "Identified using color-histogram fallback (offline mode).",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API — called by Flask route
# ─────────────────────────────────────────────────────────────────────────────

def predict_plant(image_path: str) -> dict:
    """
    Main entry point. Tries Gemini Vision first, falls back to color KNN.
    Always returns a dict compatible with the frontend displayImageResult().
    """
    try:
        # ── 1. Try Gemini Vision ──────────────────────────────────────────
        gemini_result = _predict_via_gemini(image_path)

        if gemini_result:
            crop_key   = gemini_result.get("crop_key", "unknown").lower().strip()
            plant_label = gemini_result.get(
                "plant_label",
                _CROP_LABELS.get(crop_key, crop_key.replace("_", " ").title() + " Plant")
            )
            confidence = int(gemini_result.get("confidence", 70))
            top3       = gemini_result.get("top3", [{"crop": crop_key, "confidence": confidence}])
            is_plant   = gemini_result.get("is_plant", True)
            note       = gemini_result.get("note", "")

            if not is_plant:
                return {
                    "success": False,
                    "error":   "No plant detected",
                    "message": "The uploaded image does not appear to contain a plant. "
                               "Please upload a clear photo of a plant or crop. 🌱",
                }

            # Normalize top3 format (Gemini may use slightly different keys)
            normalized_top3 = []
            for item in top3[:3]:
                c = item.get("crop") or item.get("crop_key") or crop_key
                v = item.get("confidence", 0)
                normalized_top3.append({"crop": c, "confidence": round(float(v), 1)})

            # Crop info from KB (may be empty if it's an unknown plant)
            crop_info = _get_crop_info(crop_key)

            return {
                "success":        True,
                "powered_by":     "gemini-vision",
                "predicted_crop": crop_key,
                "plant_label":    plant_label,
                "confidence":     confidence,
                "top3":           normalized_top3,
                "note":           note,
                "crop_info":      crop_info,
            }

        # ── 2. Fallback: color-histogram KNN ─────────────────────────────
        print("[INFO] Gemini Vision unavailable — using color-histogram fallback.")
        fallback = _predict_via_color_knn(image_path)
        crop_key  = fallback["crop_key"]

        return {
            "success":        True,
            "powered_by":     "color-knn-fallback",
            "predicted_crop": crop_key,
            "plant_label":    fallback["plant_label"],
            "confidence":     fallback["confidence"],
            "top3":           fallback["top3"],
            "note":           fallback["note"],
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
