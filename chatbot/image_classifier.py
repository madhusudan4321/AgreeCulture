"""
=============================================================
  AgreeCulture - Image Recognition Module
=============================================================
  This module:
  1. Accepts an uploaded plant image
  2. Extracts features using color histogram (basic, no GPU needed)
  3. Uses a simple KNN classifier trained on color signatures
  4. Returns the predicted plant name + knowledge base info

  NOTE FOR STUDENTS:
  For a real production project, you'd replace this with a
  pre-trained CNN like MobileNetV2. This version uses classical
  computer vision so it works without a GPU on your laptop.
=============================================================
"""

import numpy as np
import pickle
import json
import os
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
#  Plant color profiles (simplified simulation)
# ─────────────────────────────────────────────────────────────────────────────
# In a real system you'd train on real images. Here we simulate the classifier
# using reasonable color distributions per plant type.

# Each plant has a dominant color range [R, G, B] mean values
# that approximate the typical appearance of that plant's leaves/fruit
PLANT_COLOR_PROFILES = {
    "rice":      {"R": (100, 160), "G": (120, 180), "B": (60, 110),  "label": "Rice Plant"},
    "wheat":     {"R": (180, 220), "G": (160, 200), "B": (80,  130), "label": "Wheat Plant"},
    "maize":     {"R": (90,  140), "G": (140, 200), "B": (50,  90),  "label": "Maize Plant"},
    "tomato":    {"R": (180, 255), "G": (40,  100), "B": (30,  80),  "label": "Tomato Plant"},
    "potato":    {"R": (130, 190), "G": (120, 180), "B": (60,  120), "label": "Potato Plant"},
    "cotton":    {"R": (220, 255), "G": (220, 255), "B": (210, 255), "label": "Cotton Plant"},
    "sugarcane": {"R": (70,  120), "G": (130, 190), "B": (50,  90),  "label": "Sugarcane Plant"},
    "mango":     {"R": (60,  110), "G": (120, 180), "B": (40,  90),  "label": "Mango Tree"},
    "groundnut": {"R": (110, 160), "G": (130, 190), "B": (60,  110), "label": "Groundnut Plant"},
    "soybean":   {"R": (100, 150), "G": (140, 200), "B": (55,  100), "label": "Soybean Plant"},
}


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_color_features(image_path: str) -> np.ndarray:
    """
    Extract a feature vector from the image using:
    - Mean RGB values (3 features)
    - Standard deviation of RGB (3 features)
    - Dominant color ratio (green dominance — plants are green)

    Returns a 1D numpy array of 7 features.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))                    # Normalize size
    img_array = np.array(img, dtype=np.float32)

    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    features = [
        R.mean(), G.mean(), B.mean(),               # Mean of each channel
        R.std(),  G.std(),  B.std(),                 # Std dev (texture info)
        G.mean() / (R.mean() + B.mean() + 1e-5),    # Green dominance ratio
    ]
    return np.array(features, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Build a simple training dataset from color profiles
# ─────────────────────────────────────────────────────────────────────────────

def build_image_classifier():
    """
    Build and return a KNN classifier trained on synthetic color features.
    In production: replace this with loading a real CNN model.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    np.random.seed(42)
    X, y = [], []
    labels = list(PLANT_COLOR_PROFILES.keys())

    for crop_key, profile in PLANT_COLOR_PROFILES.items():
        # Generate 100 synthetic samples per plant
        for _ in range(100):
            r = np.random.uniform(profile["R"][0], profile["R"][1])
            g = np.random.uniform(profile["G"][0], profile["G"][1])
            b = np.random.uniform(profile["B"][0], profile["B"][1])

            # Simulate std dev (texture varies by plant type)
            r_std = np.random.uniform(20, 50)
            g_std = np.random.uniform(20, 50)
            b_std = np.random.uniform(15, 40)

            green_dom = g / (r + b + 1e-5)

            X.append([r, g, b, r_std, g_std, b_std, green_dom])
            y.append(crop_key)

    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    clf.fit(X_scaled, y)

    return clf, scaler, labels


# Build classifier once at module load (cached)
_clf, _scaler, _labels = build_image_classifier()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Prediction Function
# ─────────────────────────────────────────────────────────────────────────────

def predict_plant(image_path: str) -> dict:
    """
    Main function called by Flask app.
    Takes path to an uploaded image, returns prediction dict.
    """
    try:
        # 1. Extract features from image
        features = extract_color_features(image_path)
        features_scaled = _scaler.transform([features])

        # 2. Predict with KNN
        predicted_key = _clf.predict(features_scaled)[0]
        probabilities = _clf.predict_proba(features_scaled)[0]
        classes = _clf.classes_

        # Build confidence scores
        confidence_map = dict(zip(classes, probabilities))
        confidence = round(confidence_map[predicted_key] * 100, 1)

        # Get top 3 predictions
        sorted_preds = sorted(confidence_map.items(), key=lambda x: x[1], reverse=True)[:3]
        top3 = [{"crop": k, "confidence": round(v * 100, 1)} for k, v in sorted_preds]

        # 3. Load knowledge base info for predicted crop
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(base_dir, "data", "knowledge_base.json")
        with open(kb_path, "r") as f:
            kb = json.load(f)

        crop_info = kb["crops"].get(predicted_key, {})
        plant_label = PLANT_COLOR_PROFILES[predicted_key]["label"]

        return {
            "success": True,
            "predicted_crop": predicted_key,
            "plant_label": plant_label,
            "confidence": confidence,
            "top3": top3,
            "crop_info": {
                "season": crop_info.get("season", "N/A"),
                "soil_type": crop_info.get("soil_type", "N/A"),
                "water_requirement": crop_info.get("water_requirement", "N/A"),
                "fertilizer": crop_info.get("fertilizer", "N/A"),
                "yield": crop_info.get("yield", "N/A"),
                "tips": crop_info.get("tips", "N/A"),
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Could not process the image. Please upload a clear plant photo."
        }
