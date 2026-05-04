"""
=============================================================
  AgreeCulture - Flask Web Application
=============================================================
  Routes:
  GET  /                 → Home page
  POST /predict          → Crop recommendation (ML)
  POST /predict-insight  → Gemini AI personalized tip for predicted crop
  POST /chat             → Chatbot API (Gemini AI + rule-based fallback)
  GET  /chat-status      → Check if Gemini AI is configured
  POST /identify-plant   → Image recognition
  GET  /model-stats      → Model comparison stats (JSON)
=============================================================
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pickle
import numpy as np
import os
import json
import sys

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional; user can also set env vars manually

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chatbot.chatbot_engine import get_response          # rule-based fallback
from chatbot.image_classifier import predict_plant
from chatbot.gemini_engine import (                       # Gemini AI
    get_gemini_response,
    get_crop_insight,
    get_gemini_status,
    is_gemini_available,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App Setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────────────────────────────────────────
#  Load ML Models
# ─────────────────────────────────────────────────────────────────────────────

def load_ml_models():
    """
    Load the trained ML models, scaler, and label encoder.
    If models don't exist yet, return None values (train first!).
    """
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

    if not os.path.exists(models_dir):
        return None, None, None

    try:
        with open(os.path.join(models_dir, "random_forest.pkl"), "rb") as f:
            rf_model = pickle.load(f)
        with open(os.path.join(models_dir, "decision_tree.pkl"), "rb") as f:
            dt_model = pickle.load(f)
        with open(os.path.join(models_dir, "knn.pkl"), "rb") as f:
            knn_model = pickle.load(f)
        with open(os.path.join(models_dir, "scaler.pkl"), "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(models_dir, "label_encoder.pkl"), "rb") as f:
            label_encoder = pickle.load(f)

        return {"Random Forest": rf_model, "Decision Tree": dt_model, "KNN": knn_model}, scaler, label_encoder

    except Exception as e:
        print(f"Warning: Could not load models: {e}")
        return None, None, None


ml_models, scaler, label_encoder = load_ml_models()


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Render the main home page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_crop():
    """
    Crop Recommendation endpoint.
    Receives N, P, K, temperature, humidity, rainfall as form data.
    Returns prediction from all 3 ML models.
    """
    try:
        # Extract and validate form inputs
        features = [
            float(request.form["nitrogen"]),
            float(request.form["phosphorus"]),
            float(request.form["potassium"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["rainfall"]),
        ]
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400

    if ml_models is None:
        # Models not trained yet — return demo response
        return jsonify({
            "status": "demo",
            "message": "Models not trained yet. Run: python ml/train_model.py",
            "demo_prediction": "wheat",
            "note": "This is a demo response. Train the models first for real predictions."
        })

    # Scale the input features (same scaler used during training!)
    X = np.array([features])
    X_scaled = scaler.transform(X)

    results = {}
    votes   = {}

    for name, model in ml_models.items():
        pred_encoded = model.predict(X_scaled)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        # Get confidence (probability of the predicted class)
        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_scaled)[0]
            confidence = round(max(proba) * 100, 1)
        else:
            confidence = None

        results[name] = {"crop": pred_label, "confidence": confidence}
        votes[pred_label] = votes.get(pred_label, 0) + 1

    # Final recommendation: crop with most votes (ensemble)
    final_crop = max(votes, key=votes.get)

    # Get knowledge base info for the recommended crop
    kb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "knowledge_base.json")
    with open(kb_path, "r") as f:
        kb = json.load(f)

    crop_info = kb["crops"].get(final_crop, {})

    return jsonify({
        "status":               "success",
        "final_recommendation": final_crop,
        "model_predictions":    results,
        "crop_info": {
            "season":            crop_info.get("season", "N/A"),
            "soil_type":         crop_info.get("soil_type", "N/A"),
            "water_requirement": crop_info.get("water_requirement", "N/A"),
            "fertilizer":        crop_info.get("fertilizer", "N/A"),
            "yield":             crop_info.get("yield", "N/A"),
            "tips":              crop_info.get("tips", "N/A"),
        }
    })


@app.route("/predict-insight", methods=["POST"])
def predict_insight():
    """
    Gemini AI Insight endpoint.
    Call this after /predict to get a personalized Gemini tip for the farmer.
    Receives JSON: { crop, nitrogen, phosphorus, potassium, temperature, humidity, rainfall }
    """
    data = request.get_json()
    if not data or "crop" not in data:
        return jsonify({"error": "No crop provided"}), 400

    crop_name = data.get("crop", "")
    features  = {
        "nitrogen":    data.get("nitrogen"),
        "phosphorus":  data.get("phosphorus"),
        "potassium":   data.get("potassium"),
        "temperature": data.get("temperature"),
        "humidity":    data.get("humidity"),
        "rainfall":    data.get("rainfall"),
    }

    insight = get_crop_insight(crop_name, features)

    if insight:
        return jsonify({"status": "success", "insight": insight, "powered_by": "gemini"})
    else:
        return jsonify({"status": "unavailable", "insight": None})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chatbot API endpoint.
    Uses Google Gemini AI (primary) with rule-based fallback.
    Receives JSON: { "message": "...", "session_id": "..." (optional) }
    """
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "No message provided"}), 400

    user_message = data["message"].strip()
    session_id   = data.get("session_id", "default")

    # ── Unified AI engine: Gemini → OpenRouter → local fallback ──────────────
    result = get_gemini_response(user_message, session_id=session_id)

    if result["success"]:
        return jsonify({
            "status":       "success",
            "user_message": user_message,
            "bot_response": result["response"],
            "powered_by":   result["powered_by"],   # "gemini" | "openrouter"
        })

    # All AI providers exhausted — use rule-based local fallback
    bot_response = get_response(user_message)
    return jsonify({
        "status":       "success",
        "user_message": user_message,
        "bot_response": bot_response,
        "powered_by":   "fallback",
    })


@app.route("/chat-status")
def chat_status():
    """Return whether Gemini AI is configured and ready."""
    status = get_gemini_status()
    return jsonify(status)


@app.route("/identify-plant", methods=["POST"])
def identify_plant():
    """
    Image Recognition endpoint.
    Accepts an uploaded image file, returns predicted plant name + info.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed. Use: PNG, JPG, JPEG"}), 400

    # Save the uploaded file
    from werkzeug.utils import secure_filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run prediction
    result = predict_plant(filepath)
    result["image_url"] = f"/static/uploads/{filename}"

    return jsonify(result)


@app.route("/model-stats")
def model_stats():
    """Return model comparison stats from JSON file."""
    stats_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "model_comparison.json")

    if not os.path.exists(stats_path):
        # Return demo stats if not trained (values match what training actually produces)
        return jsonify({
            "Random Forest": {"accuracy": 86.75},
            "Decision Tree": {"accuracy": 80.75},
            "KNN":           {"accuracy": 86.25},
            "note": "Demo values — run ml/train_model.py for real stats"
        })

    with open(stats_path, "r") as f:
        stats = json.load(f)

    return jsonify(stats)


# ─────────────────────────────────────────────────────────────────────────────
#  Error Handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    """Return a clean JSON 404 for API calls, or redirect home for browser."""
    if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
        return jsonify({"error": "Not found", "status": 404}), 404
    return redirect(url_for("index"))


@app.errorhandler(413)
def file_too_large(e):
    """Handle uploads exceeding MAX_CONTENT_LENGTH."""
    return jsonify({"error": "File too large. Maximum size is 5 MB."}), 413


# ─────────────────────────────────────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  AgreeCulture Smart Farming Assistant")
    print("="*55)
    print("  Open in browser: http://127.0.0.1:5000")

    gemini_key    = os.environ.get("GEMINI_API_KEY", "")
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print("  [AI-1] Gemini:     ENABLED  (primary)")
    else:
        print("  [AI-1] Gemini:     NOT configured")

    if openrouter_key and openrouter_key != "your_openrouter_api_key_here":
        print("  [AI-2] OpenRouter: ENABLED  (fallback)")
    else:
        print("  [AI-2] OpenRouter: NOT configured (add OPENROUTER_API_KEY to .env)")

    print("  Press Ctrl+C to stop the server")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)

