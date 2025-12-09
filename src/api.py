import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# --- Configuration du Monitoring ---
LOG_FILE = 'monitoring_logs/prediction_log.csv' # Le fichier où nous allons stocker les données entrantes

# Assurez-vous que le dossier de logs existe
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# Définition des chemins absolus (comme corrigé précédemment)
MODEL_PATH = '/app/models/model_champion.pkl'
SCALER_PATH = '/app/models/scaler.pkl'

# Charger le modèle et le scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modèle et Scaler chargés avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle ou du scaler: {e}")
    # Si le chargement échoue, l'API ne fonctionnera pas
    model = None
    scaler = None
    
# Initialisation de Flask
app = Flask(__name__)

# Fonction utilitaire pour enregistrer les données de production
def log_prediction(data_dict, prediction, probability):
    """Enregistre les données d'entrée et le résultat dans un fichier CSV."""
    # Convertit les données JSON en DataFrame
    df_log = pd.DataFrame([data_dict])
    
    # Ajoute les résultats de la prédiction
    df_log['prediction'] = prediction
    df_log['probability_churn'] = probability
    df_log['timestamp'] = pd.to_datetime('now')
    
    # Écrit dans le fichier log
    if not os.path.exists(LOG_FILE):
        # Crée l'en-tête si le fichier n'existe pas
        df_log.to_csv(LOG_FILE, index=False)
    else:
        # Ajoute les données sans l'en-tête
        df_log.to_csv(LOG_FILE, mode='a', header=False, index=False)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Modèle non chargé, l'API n'est pas prête."}), 503
    
    try:
        json_data = request.json
        data = pd.DataFrame([json_data])
        
        # 1. Prétraitement des données (Assurez-vous que le scaler est appliqué aux bonnes colonnes)
        # Assurez-vous d'identifier les colonnes numériques qui ont été scalées
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        data[numeric_cols] = scaler.transform(data[numeric_cols])

        # 2. Prédiction
        probability = model.predict_proba(data)[:, 1][0]
        prediction = int(probability >= 0.5)

        response = {
            "prediction": prediction,
            "probability_churn": float(probability)
        }
        
        # 3. Enregistrement des données de production (Monitoring)
        log_prediction(json_data, prediction, probability)
        
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

# Le `CMD` Gunicorn dans le Dockerfile est toujours la méthode recommandée pour la production.
# CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api:app"]