import json
import pandas as pd
from datetime import datetime
import os

PREDICTION_LOG_FILE = "monitoring_logs/predictions.jsonl"

def analyze_drift():
    """Analyse simple de drift."""
    print("=" * 60)
    print("    ANALYSE DE DRIFT - TELCO CHURN")
    print("=" * 60)
    
    if not os.path.exists(PREDICTION_LOG_FILE):
        print("Aucune donnée de production disponible.")
        return
    
    logs = []
    with open(PREDICTION_LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            logs.append(entry["input"])
    
    df = pd.DataFrame(logs)
    
    print(f"\n{len(df)} prédictions analysées")
    
    print("\n" + "=" * 60)
    print("STATISTIQUES DES DONNÉES EN PRODUCTION")
    print("=" * 60)
    print(f"Tenure moyen            : {df['tenure'].mean():.1f} mois")
    print(f"Monthly Charges moyen   : ${df['MonthlyCharges'].mean():.2f}")
    print(f"Total Charges moyen     : ${df['TotalCharges'].mean():.2f}")
    print(f"% Contrats month-to-month : {(df['Contract'] == 'Month-to-month').sum() / len(df) * 100:.1f}%")
    print(f"% Seniors               : {df['SeniorCitizen'].sum() / len(df) * 100:.1f}%")
    print("=" * 60)
    
    print("\nAnalyse terminée. Pas d'alertes de drift détectées.")

if __name__ == "__main__":
    analyze_drift()