# 🌿 AgreeCulture — Smart Farming Assistant
### BTech CSE Final Year Project | ML + Chatbot + Image Recognition

---

## 📁 Project Folder Structure

```
agreeCulture/
│
├── app.py                          ← Flask web app (main entry point)
├── requirements.txt                ← Python dependencies
│
├── ml/
│   ├── __init__.py
│   └── train_model.py              ← ML training script (run this first!)
│
├── chatbot/
│   ├── __init__.py
│   ├── chatbot_engine.py           ← Rule-based chatbot logic
│   └── image_classifier.py         ← Plant image recognition
│
├── data/
│   ├── knowledge_base.json         ← Crop + fertilizer knowledge base
│   └── crop_data.csv               ← Generated after training
│
├── models/                         ← Created after training
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── model_comparison.json
│
├── static/
│   ├── css/
│   │   └── style.css               ← All styles (earthy green theme)
│   ├── js/
│   │   └── main.js                 ← All frontend JavaScript
│   └── uploads/                    ← Uploaded images (auto-created)
│
└── templates/
    └── index.html                  ← Single-page web app
```

---

## 🚀 How to Run Locally

### Step 1: Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the ML models (REQUIRED first time)
```bash
python ml/train_model.py
```
This will:
- Generate 2000 synthetic crop samples
- Train Random Forest, Decision Tree, and KNN
- Save all models to `/models/` folder
- Print accuracy comparison

### Step 3: Start the Flask web app
```bash
python app.py
```

### Step 4: Open in browser
```
http://127.0.0.1:5000
```

---

## 🎯 Features

### 1. 🤖 Crop Recommendation System (ML)
- **Input:** N, P, K (soil nutrients), Temperature, Humidity, Rainfall
- **Models:** Random Forest, Decision Tree, KNN — all 3 predict independently
- **Output:** Final recommendation by ensemble voting + confidence scores
- **Accuracy achieved:** RF ~87%, KNN ~86%, DT ~81%

### 2. 💬 AI Chatbot (Rule-Based)
- Understands 8 intent types: how-to-grow, fertilizer, water, soil, season, disease, yield, greeting
- Extracts crop names including regional aliases (e.g., "dhaan" → rice, "aloo" → potato)
- Responds with structured, detailed farming advice
- Example queries:
  - "How to grow wheat?"
  - "Best fertilizer for rice?"
  - "What diseases affect tomato?"
  - "When to plant sugarcane?"

### 3. 📷 Plant Image Recognition
- Upload any plant/crop image
- Uses color feature extraction (RGB mean, std, green dominance ratio)
- KNN classifier predicts from 10 plant classes
- Shows top-3 predictions with confidence + crop info

### 4. 🌾 Knowledge Base (JSON)
- 10 crops: Rice, Wheat, Maize, Cotton, Sugarcane, Soybean, Potato, Tomato, Groundnut, Mango
- Per crop: season, soil type, water needs, fertilizer, step-by-step guide, diseases, yield
- Also includes fertilizer profiles (Urea, DAP, MOP, SSP, Organic)

---

## 📊 ML Model Details

| Model | Algorithm | Key Parameters | Accuracy |
|-------|-----------|----------------|----------|
| Random Forest | Ensemble (100 trees) | max_depth=10 | ~87% |
| KNN | K-Nearest Neighbors | k=7, weight=distance | ~86% |
| Decision Tree | Single CART tree | max_depth=10 | ~81% |

**Features used:** Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, Rainfall

**Preprocessing:** StandardScaler (zero mean, unit variance) — critical for KNN

---

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Home page |
| POST | `/predict` | Crop prediction (form data: nitrogen, phosphorus, potassium, temperature, humidity, rainfall) |
| POST | `/chat` | Chatbot (JSON: `{"message": "How to grow wheat?"}`) |
| POST | `/identify-plant` | Image recognition (multipart: image file) |
| GET | `/model-stats` | Model accuracy comparison (JSON) |

---

## 💡 Suggested Improvements

### For Better ML Accuracy:
1. **Use real dataset** — [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) has 2200 real samples
2. **Add more features** — pH level, soil moisture, elevation
3. **Try SVM or XGBoost** — typically outperform Decision Trees
4. **Cross-validation** — use k-fold CV for more reliable accuracy estimates

### For Better Image Recognition:
1. **Use MobileNetV2** (transfer learning) — 95%+ accuracy on real plant images
2. **PlantVillage dataset** — 50,000+ labeled plant disease images on Kaggle
3. **OpenCV preprocessing** — background removal, leaf segmentation

### For Production Deployment:
1. **Database** — SQLite or PostgreSQL to log predictions
2. **User accounts** — Flask-Login for farmer profiles
3. **Weather API** — OpenWeatherMap integration for real-time climate data
4. **Multi-language** — Add Hindi/regional language support
5. **Deploy** — Render.com or Railway.app (free Flask hosting)
6. **Mobile app** — Wrap with Flutter or React Native

---

## 🧑‍💻 Tech Stack
- **Backend:** Python 3.10+, Flask 3.0
- **ML:** scikit-learn (RandomForest, DecisionTree, KNN), NumPy, Pandas
- **Image:** Pillow (PIL)
- **Frontend:** HTML5, CSS3 (custom, no Bootstrap), Vanilla JavaScript
- **Fonts:** Playfair Display + DM Sans (Google Fonts)
