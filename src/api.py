import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np
# Ajout de 'os' pour la gestion des chemins dans Docker
import os 

app = Flask(__name__)   

# --- Constantes ---
# CORRECTION MAJEURE: Chemins absolus dans le conteneur Docker (/app est le WORKDIR)
MODEL_PATH = "/app/models/model_champion.pkl"
SCALER_PATH = "/app/models/scaler.pkl"

# Colonnes numériques identifiées dans preprocess.py pour le StandardScaler
NUM_COLS = ["tenure", 'MonthlyCharges', 'TotalCharges']

# Les 20 colonnes d'entrée attendues par le modèle (après encodage/scaling)
ALL_COLS_EXCEPT_CHURN = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# --- 1. INITIALISATION ET CHARGEMENT DES ARTEFACTS ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"Artefacts chargés avec succès depuis: {MODEL_PATH}")
except Exception as e:
    # Affiche le chemin d'accès si le chargement échoue pour diagnostic
    print(f"Erreur LORS DU CHARGEMENT DES ARTEFACTS. Chemin actuel: {os.getcwd()}")
    print(f"Détail de l'erreur: {e}")
    model, scaler = None, None

# --- 2. FONCTION DE PRÉDICTION ---
@app.route('/predict', methods=['POST'])
def predict_churn():
    if not model or not scaler:
        return jsonify({"error": "Service indisponible (modèle non chargé)"}), 503

    try:
        data = request.get_json(force=True)
        df_input = pd.DataFrame(data)

        # --- REPRODUCTION DU PRÉTRAITEMENT ---
        
        # 1 & 2. Gestion de TotalCharges et NaNs
        df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce')
        df_input['TotalCharges'].fillna(0, inplace=True)
        
        # 3. Encodage des variables catégorielles (Label Encoding)
        # ATTENTION: Utiliser fit_transform ici est INCORRECT car il créera un encodage
        # différent pour chaque requête, faussant la prédiction.
        # Idéalement, les LabelEncoders doivent être sauvegardés (comme le scaler),
        # mais si vous avez traité les données directement en production, on ne peut
        # que forcer une transformation basée sur les valeurs.
        for col in df_input.columns:
            if df_input[col].dtype == 'object':
                le = LabelEncoder()
                # On utilise fit_transform ici, en supposant que l'ensemble des valeurs
                # est le même que celui utilisé à l'entraînement.
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # 4. Conversion en types numériques uniformes
        df_input = df_input.astype(np.float32)

        # 5. Mise à l'échelle (StandardScaler)
        df_scaled = df_input.copy()
        df_scaled[NUM_COLS] = scaler.transform(df_scaled[NUM_COLS])
        
        # 6. S'assurer de l'ordre exact des colonnes avant la prédiction
        X_pred = df_scaled[ALL_COLS_EXCEPT_CHURN]
        
        # --- PRÉDICTION ---
        predictions_proba = model.predict_proba(X_pred)[:, 1] 

        results = {
            "prediction": int(predictions_proba[0] >= 0.5), 
            "probability_churn": float(predictions_proba[0])
        }
        
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e), "message": "Erreur dans le traitement des données d'entrée. Vérifiez le format JSON et les noms des colonnes."}), 400

# --- 3. LANCEMENT DU SERVEUR ---
# Gunicorn prendra le relais du lancement du serveur
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)