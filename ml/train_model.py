"""
=============================================================
  AgreeCulture - Crop Recommendation ML Model Training
=============================================================
  This script:
  1. Generates a synthetic crop dataset (since we can't download online)
  2. Preprocesses the data (scaling, encoding)
  3. Trains 3 models: Random Forest, Decision Tree, KNN
  4. Evaluates and compares all models
  5. Saves the best model to disk for use in the Flask app
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import json

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1: Generate synthetic but realistic crop dataset
# ─────────────────────────────────────────────────────────────────────────────
# Each crop has realistic ranges for: N, P, K, temperature, humidity, rainfall
# These values are based on agricultural research references

def generate_crop_data(samples_per_crop=200):
    """
    Generate a synthetic dataset with realistic agronomic ranges.
    Returns a pandas DataFrame with columns: N, P, K, temperature, humidity, rainfall, label
    """

    # Define realistic parameter ranges for each crop
    # Format: (N_mean, N_std, P_mean, P_std, K_mean, K_std,
    #           temp_mean, temp_std, humidity_mean, humidity_std,
    #           rainfall_mean, rainfall_std)
    crop_params = {
        "rice":      (80, 15, 45, 10, 40, 10, 25, 2, 82, 5, 200, 30),
        "wheat":     (100, 20, 40, 10, 42, 10, 20, 3, 65, 8, 80,  20),
        "maize":     (85, 15, 58, 12, 43, 10, 25, 3, 65, 8, 100, 25),
        "cotton":    (118, 20, 45, 10, 43, 10, 25, 3, 80, 6, 80,  20),
        "sugarcane": (87, 15, 46, 10, 50, 10, 27, 2, 75, 6, 150, 30),
        "soybean":   (43, 10, 70, 12, 40, 10, 29, 3, 65, 8, 100, 25),
        "potato":    (62, 12, 60, 12, 100,15, 19, 3, 80, 6, 80,  20),
        "tomato":    (18, 8,  22, 8,  120,15, 26, 3, 82, 5, 100, 20),
        "groundnut": (22, 8,  40, 10, 50, 10, 28, 3, 65, 8, 80,  20),
        "mango":     (14, 5,  25, 8,  39, 8,  30, 2, 50, 8, 100, 25),
    }

    all_data = []
    np.random.seed(42)  # For reproducibility

    for crop, params in crop_params.items():
        N_m, N_s, P_m, P_s, K_m, K_s, T_m, T_s, H_m, H_s, R_m, R_s = params

        for _ in range(samples_per_crop):
            row = {
                "N":           max(0, round(np.random.normal(N_m, N_s), 1)),
                "P":           max(0, round(np.random.normal(P_m, P_s), 1)),
                "K":           max(0, round(np.random.normal(K_m, K_s), 1)),
                "temperature": round(np.random.normal(T_m, T_s), 1),
                "humidity":    min(100, max(0, round(np.random.normal(H_m, H_s), 1))),
                "rainfall":    max(0, round(np.random.normal(R_m, R_s), 1)),
                "label":       crop
            }
            all_data.append(row)

    df = pd.DataFrame(all_data)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_data(df):
    """
    Encode labels and scale features.
    Returns: X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    print("\n📊 Dataset Info:")
    print(f"   Total samples: {len(df)}")
    print(f"   Crops: {df['label'].unique().tolist()}")
    print(f"   Features: {[c for c in df.columns if c != 'label']}")
    print(f"\n   Class distribution:\n{df['label'].value_counts().to_string()}")

    # Encode crop labels to numbers (e.g., "rice" → 0, "wheat" → 1)
    label_encoder = LabelEncoder()
    df["label_encoded"] = label_encoder.fit_transform(df["label"])

    # Separate features and target
    feature_cols = ["N", "P", "K", "temperature", "humidity", "rainfall"]
    X = df[feature_cols].values
    y = df["label_encoded"].values

    # Split into train (80%) and test (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features: StandardScaler ensures mean=0, std=1
    # This is important for KNN (distance-based) and helps other models too
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)   # Fit on train, transform train
    X_test = scaler.transform(X_test)         # Only transform test (no fit!)

    print(f"\n✅ Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, scaler, label_encoder


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3: Train all 3 models
# ─────────────────────────────────────────────────────────────────────────────

def train_models(X_train, y_train):
    """
    Train Random Forest, Decision Tree, and KNN models.
    Returns a dict of trained model objects.
    """
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,      # 100 decision trees in the forest
            max_depth=10,          # Prevent overfitting
            random_state=42,
            n_jobs=-1              # Use all CPU cores
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,          # Limit depth to prevent overfitting
            min_samples_split=5,   # At least 5 samples needed to split
            random_state=42
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,         # Consider 7 nearest neighbors
            weights="distance",    # Closer neighbors have more influence
            metric="euclidean"
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"\n🌱 Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"   ✅ {name} trained successfully!")

    return trained


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4: Evaluate and compare models
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_models(models, X_test, y_test, label_encoder):
    """
    Evaluate all models on test data. Print accuracy and classification report.
    Returns the name of the best model and results dict.
    """
    results = {}
    best_model_name = None
    best_accuracy = 0

    print("\n" + "="*60)
    print("  📈 MODEL EVALUATION RESULTS")
    print("="*60)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": round(accuracy * 100, 2),
            "model": model
        }

        print(f"\n🔷 {name}")
        print(f"   Accuracy: {accuracy * 100:.2f}%")

        # Detailed per-crop metrics
        report = classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        results[name]["report"] = report

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name

    print("\n" + "="*60)
    print(f"  🏆 Best Model: {best_model_name} ({best_accuracy*100:.2f}% accuracy)")
    print("="*60)

    return best_model_name, results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5: Save models and preprocessing objects
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(models, scaler, label_encoder, results, output_dir="models"):
    """
    Save all trained models, scaler, and encoder to disk using pickle.
    Also save a comparison summary as JSON.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save each model
    for name, model in models.items():
        filename = name.lower().replace(" ", "_") + ".pkl"
        path = os.path.join(output_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"   💾 Saved: {filename}")

    # Save the scaler (needed to scale new input before prediction)
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("   💾 Saved: scaler.pkl")

    # Save the label encoder (needed to decode prediction back to crop name)
    with open(os.path.join(output_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    print("   💾 Saved: label_encoder.pkl")

    # Save model comparison summary for the frontend to display
    summary = {
        name: {"accuracy": info["accuracy"]}
        for name, info in results.items()
    }
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("   💾 Saved: model_comparison.json")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN: Run the full pipeline
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("="*60)
    print("  🌾 AgreeCulture - ML Model Training Pipeline")
    print("="*60)

    # Resolve absolute paths so the script works from ANY directory
    ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR  = os.path.join(ROOT_DIR, "data")
    MODEL_DIR = os.path.join(ROOT_DIR, "models")
    os.makedirs(DATA_DIR,  exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Step 1: Generate data
    print("\n📦 Step 1: Generating synthetic crop dataset...")
    df = generate_crop_data(samples_per_crop=200)
    csv_path = os.path.join(DATA_DIR, "crop_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   ✅ Dataset saved to {csv_path} ({len(df)} rows)")

    # Step 2: Preprocess
    print("\n⚙️  Step 2: Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(df)

    # Step 3: Train
    print("\n🚀 Step 3: Training models...")
    trained_models = train_models(X_train, y_train)

    # Step 4: Evaluate
    print("\n📊 Step 4: Evaluating models...")
    best_name, results = evaluate_models(trained_models, X_test, y_test, label_encoder)

    # Step 5: Save
    print("\n💾 Step 5: Saving trained models...")
    save_artifacts(trained_models, scaler, label_encoder, results, output_dir=MODEL_DIR)

    print("\n✅ Training complete! All models saved to /models directory.")
    print("   You can now run: python app.py")
