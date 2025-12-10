import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# ================================
# CONSTANTES
# ================================
MODEL_PATH = "/app/models/model_champion.pkl"
SCALER_PATH = "/app/models/scaler.pkl"
ENCODERS_PATH = "/app/models/encoders.pkl"   # fichier à générer pendant preprocessing

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

ALL_MODEL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges"
]

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
    print("\n>>> ERREUR : Impossible de charger les artefacts !")
    print(f"Chemin actuel : {os.getcwd()}")
    print(f"Détail : {e}\n")
    model, scaler, encoders = None, None, None


# ================================
# FONCTION DE PRÉTRAITEMENT
# ================================
def preprocess_input(json_data):
    """
    Prépare les données brutes pour le modèle.
    """
    df = pd.DataFrame([json_data])

    # 1. Convertir TotalCharges en numérique
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # 2. Encodage : appliquer les encoders sauvegardés
    if encoders is None:
        raise ValueError("Encoders manquants. Fournissez encoders.pkl dans /models.")

    for col, mapping in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # 3. Conversion vers float
    df = df.astype(np.float32)

    # 4. Scaling
    df_scaled = df.copy()
    df_scaled[NUM_COLS] = scaler.transform(df_scaled[NUM_COLS])

    # 5. Sélection et ordre exact des colonnes
    df_scaled = df_scaled[ALL_MODEL_COLS]

    return df_scaled


# ================================
# ROUTE : PREDICTION
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Service indisponible (modèle non chargé)"}), 503

    try:
        input_json = request.get_json(force=True)

        processed = preprocess_input(input_json)

        prob = model.predict_proba(processed)[:, 1][0]
        prediction = int(prob >= 0.5)

        return jsonify({
            "prediction": prediction,
            "probability_churn": float(prob)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Erreur lors du prétraitement. Vérifiez vos colonnes."
        }), 400


# ================================
# LANCEMENT LOCAL (Gunicorn en prod)
# ================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
