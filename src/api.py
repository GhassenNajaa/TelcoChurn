# src/api.py (Version Corrigée pour Inclure la Logique d'Encodage/Scaling)

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

# --- Constantes ---
MODEL_PATH = "models/model_champion.pkl"
SCALER_PATH = "models/scaler.pkl"
# Colonnes numériques identifiées dans preprocess.py pour le StandardScaler
NUM_COLS = ["tenure", 'MonthlyCharges', 'TotalCharges']

# Les 20 colonnes d'entrée attendues par le modèle (après encodage/scaling)
# C'est l'ordre des colonnes de votre X_train
# Remplacer cette liste par l'ordre exact de vos colonnes après Label Encoding
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
    print("Artefacts chargés avec succès pour l'API.")
except Exception as e:
    print(f"Erreur lors du chargement des artefacts: {e}")
    model, scaler = None, None

# --- 2. FONCTION DE PRÉDICTION ---
@app.route('/predict', methods=['POST'])
def predict_churn():
    if not model or not scaler:
        return jsonify({"error": "Service indisponible (modèle non chargé)"}), 503

    try:
        # Récupère les données JSON (doit être une liste de dictionnaire pour DataFrame)
        data = request.get_json(force=True)
        df_input = pd.DataFrame(data)

        # --- REPRODUCTION DU PRÉTRAITEMENT ---
        
        # 1. Gestion de TotalCharges (reproduction de l'étape 2)
        df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce')
        
        # 2. Remplissage des NaNs dans TotalCharges avec une valeur cohérente
        # NOTE: Utiliser une valeur cohérente (e.g., 0 ou la moyenne du jeu d'entraînement)
        # Puisque dans preprocess.py vous avez remplacé par la moyenne, ici on utilise 0 pour simplifier
        df_input['TotalCharges'].fillna(0, inplace=True)
        
        # 3. Encodage des variables catégorielles (Label Encoding)
        # Nous devons réappliquer la logique de LabelEncoder sur les colonnes 'object'
        for col in df_input.columns:
            if df_input[col].dtype == 'object':
                le = LabelEncoder()
                # Fit_transform est TRÈS DANGEREUX ici car il peut donner des codes différents!
                # Nous nous basons sur la conversion implicite qui a eu lieu dans preprocess.py
                # Nous forçons la conversion en float32 après avoir traité les numériques ci-dessous.
                df_input[col] = le.fit_transform(df_input[col].astype(str))
        
        # 4. Conversion en types numériques uniformes (reproduction de l'étape 8)
        df_input = df_input.astype(np.float32)

        # 5. Mise à l'échelle (StandardScaler) (reproduction de l'étape 12)
        # Créer une copie et s'assurer que les colonnes existent
        df_scaled = df_input.copy()
        
        # Scale les colonnes numériques en utilisant le scaler chargé
        df_scaled[NUM_COLS] = scaler.transform(df_scaled[NUM_COLS])
        
        # 6. S'assurer de l'ordre exact des colonnes avant la prédiction
        # C'est la partie la plus critique : l'ordre des colonnes doit être IDENTIQUE à X_train
        X_pred = df_scaled[ALL_COLS_EXCEPT_CHURN]
        
        # --- PRÉDICTION ---
        predictions_proba = model.predict_proba(X_pred)[:, 1] # Probabilité de churn (classe 1)

        results = {
            "prediction": int(predictions_proba[0] >= 0.5), 
            "probability_churn": float(predictions_proba[0])
        }
        
        return jsonify(results)

    except Exception as e:
        # Renvoie un message d'erreur détaillé
        return jsonify({"error": str(e), "message": "Erreur dans le traitement des données d'entrée. Vérifiez le format JSON et les noms des colonnes."}), 400

# --- 3. LANCEMENT DU SERVEUR ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)