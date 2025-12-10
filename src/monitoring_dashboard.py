import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration
PREDICTION_LOG_FILE = "monitoring_logs/predictions.jsonl"
OUTPUT_DIR = "monitoring_logs/reports"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all_predictions():
    """Charge tous les logs de prédictions."""
    if not os.path.exists(PREDICTION_LOG_FILE):
        print(f"Fichier de logs introuvable : {PREDICTION_LOG_FILE}")
        return pd.DataFrame()
    
    logs = []
    with open(PREDICTION_LOG_FILE, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            logs.append({
                "timestamp": datetime.fromisoformat(entry["timestamp"]),
                "prediction": entry["prediction"],
                "probability_churn": entry["probability_churn"],
                "response_time_ms": entry["response_time_ms"]
            })
    
    df = pd.DataFrame(logs)
    if not df.empty:
        df = df.sort_values("timestamp")
    
    return df

def generate_dashboard():
    """Génère un dashboard de monitoring avec graphiques."""
    print("=" * 60)
    print("    DASHBOARD DE MONITORING - TELCO CHURN API")
    print("=" * 60)
    
    df = load_all_predictions()
    
    if df.empty:
        print(" Aucune prédiction enregistrée.")
        return
    
    print(f"\n{len(df)} prédictions chargées")
    print(f"📅 Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Créer une figure avec 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Dashboard de Monitoring - Telco Churn API', fontsize=16, fontweight='bold')
    
    # 1. Distribution des prédictions
    ax1 = axes[0, 0]
    prediction_counts = df['prediction'].value_counts().sort_index()
    colors = ['green', 'red']
    ax1.bar(prediction_counts.index, prediction_counts.values, color=colors, alpha=0.7)
    ax1.set_xlabel('Prédiction')
    ax1.set_ylabel('Nombre')
    ax1.set_title('Distribution des Prédictions')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['No Churn', 'Churn'])
    ax1.grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(prediction_counts.values):
        ax1.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Distribution des probabilités de churn
    ax2 = axes[0, 1]
    ax2.hist(df['probability_churn'], bins=10, color='blue', alpha=0.7, edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Seuil de décision')
    ax2.set_xlabel('Probabilité de Churn')
    ax2.set_ylabel('Fréquence')
    ax2.set_title('Distribution des Probabilités de Churn')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Temps de réponse de l'API
    ax3 = axes[1, 0]
    ax3.plot(range(len(df)), df['response_time_ms'], marker='o', linestyle='-', alpha=0.6)
    ax3.axhline(df['response_time_ms'].mean(), color='red', linestyle='--', 
                label=f'Moyenne: {df["response_time_ms"].mean():.2f} ms')
    ax3.set_xlabel('Requête #')
    ax3.set_ylabel('Temps de Réponse (ms)')
    ax3.set_title('Performance de l\'API')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. Top 5 clients à risque
    ax4 = axes[1, 1]
    top_risk = df.nlargest(5, 'probability_churn')
    ax4.barh(range(len(top_risk)), top_risk['probability_churn'], color='orange', alpha=0.7)
    ax4.set_yticks(range(len(top_risk)))
    ax4.set_yticklabels([f'Client #{i+1}' for i in range(len(top_risk))])
    ax4.set_xlabel('Probabilité de Churn')
    ax4.set_title('Top 5 Clients à Risque')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarder le dashboard
    output_file = os.path.join(OUTPUT_DIR, f"dashboard_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nDashboard sauvegardé : {output_file}")
    
    # Afficher les statistiques
    print("\n" + "=" * 60)
    print("STATISTIQUES")
    print("=" * 60)
    print(f"Total de prédictions         : {len(df)}")
    print(f"Prédictions 'Churn'          : {(df['prediction'] == 1).sum()} ({(df['prediction'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"Prédictions 'No Churn'       : {(df['prediction'] == 0).sum()} ({(df['prediction'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"Probabilité moyenne de churn : {df['probability_churn'].mean():.3f}")
    print(f"Temps de réponse moyen       : {df['response_time_ms'].mean():.2f} ms")
    print(f"Temps de réponse max         : {df['response_time_ms'].max():.2f} ms")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    generate_dashboard()