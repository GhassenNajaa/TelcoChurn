import os
import joblib
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from flask import Flask, request, jsonify

app = Flask(__name__)

# ================================
# CONSTANTES
# ================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model_champion.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "encoders.pkl")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "monitoring_logs")

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

ALL_MODEL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

# ================================
# CRÉATION DU DOSSIER DE LOGS
# ================================
os.makedirs(LOG_DIR, exist_ok=True)
PREDICTION_LOG_FILE = os.path.join(LOG_DIR, "predictions.jsonl")
METRICS_LOG_FILE = os.path.join(LOG_DIR, "metrics.jsonl")

# ================================
# CHARGEMENT DES ARTEFACTS
# ================================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
    else:
        encoders = None

    print(">>> Modèle, scaler et encoders chargés avec succès.")
except Exception as e:
    print(f"\n>>> ERREUR : Impossible de charger les artefacts : {e}")
    model, scaler, encoders = None, None, None


# ================================
# FONCTION DE LOGGING
# ================================
def log_prediction(input_data, prediction, probability, response_time):
    """Logger chaque prédiction dans un fichier JSONL."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "prediction": int(prediction),
        "probability_churn": float(probability),
        "response_time_ms": round(response_time * 1000, 2)
    }
    
    try:
        with open(PREDICTION_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Erreur lors du logging : {e}")


def log_metrics(metric_name, value):
    """Logger des métriques système."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "metric": metric_name,
        "value": value
    }
    
    try:
        with open(METRICS_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Erreur lors du logging des métriques : {e}")


# ================================
# FONCTION DE PRÉTRAITEMENT
# ================================
def preprocess_input(json_data):
    """Prépare les données brutes pour le modèle."""
    df = pd.DataFrame([json_data])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    if encoders is None:
        raise ValueError("Encoders manquants.")

    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    df = df.astype(np.float32)

    df_scaled = df.copy()
    df_scaled[NUM_COLS] = scaler.transform(df_scaled[NUM_COLS])

    df_scaled = df_scaled[ALL_MODEL_COLS]

    return df_scaled


# ================================
# ROUTES
# ================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "version": "v1.0",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": encoders is not None
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint pour consulter les métriques basiques."""
    try:
        prediction_count = 0
        if os.path.exists(PREDICTION_LOG_FILE):
            with open(PREDICTION_LOG_FILE, "r") as f:
                prediction_count = sum(1 for _ in f)
        
        return jsonify({
            "total_predictions": prediction_count,
            "log_file": PREDICTION_LOG_FILE,
            "metrics_file": METRICS_LOG_FILE
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        log_metrics("error_model_unavailable", 1)
        return jsonify({"error": "Service indisponible (modèle non chargé)"}), 503

    start_time = time.time()

    try:
        input_json = request.get_json(force=True)
        processed = preprocess_input(input_json)
        prob = model.predict_proba(processed)[:, 1][0]
        prediction = int(prob >= 0.5)
        response_time = time.time() - start_time

        log_prediction(input_json, prediction, prob, response_time)
        log_metrics("prediction_success", 1)

        return jsonify({
            "prediction": prediction,
            "probability_churn": float(prob),
            "response_time_ms": round(response_time * 1000, 2)
        })

    except Exception as e:
        response_time = time.time() - start_time
        log_metrics("prediction_error", 1)
        
        return jsonify({
            "error": str(e),
            "message": "Erreur lors du prétraitement.",
            "response_time_ms": round(response_time * 1000, 2)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)