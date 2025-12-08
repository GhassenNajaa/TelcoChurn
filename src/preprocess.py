# src/preprocess.py (Version Totalement Corrigée)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib 

# --- Constantes ---
RAW_DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_DATA_DIR = 'data/processed'
SCALER_PATH = 'models/scaler.pkl' 
TEST_SIZE = 0.3
RANDOM_STATE = 40

def run_preprocessing():
    """Charge, nettoie, transforme, scale, split et sauvegarde les données."""
    
    # 1. Chargement
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(['customerID'], axis=1)
    
    # 2. Gestion de TotalCharges AVANT tout autre traitement
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # 3. Suppression des lignes avec tenure 0
    df = df[df['tenure'] != 0].copy()  # Utiliser .copy() pour éviter les warnings
    
    # 4. Remplissage des NaNs dans TotalCharges avec la moyenne
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    
    # 5. Transformation de SeniorCitizen (si c'est une chaîne, sinon skipper)
    if df["SeniorCitizen"].dtype == 'object':
        df["SeniorCitizen"] = df["SeniorCitizen"].map({"No": 0, "Yes": 1})
    
    # 6. Encodage des variables catégorielles
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))  # Convertir en string d'abord
    
    # 7. Vérification et suppression finale des NaN (CRITIQUE)
    print(f"NaN avant suppression finale: {df.isnull().sum().sum()}")
    df.dropna(inplace=True)
    print(f"NaN après suppression finale: {df.isnull().sum().sum()}")
    print(f"Shape finale: {df.shape}")
    
    # 8. Conversion en types numériques uniformes
    df = df.astype(np.float32)
    
    # 9. Séparation X/y
    X = df.drop(columns=['Churn'])
    y = df['Churn'].values.astype(int)
    
    # 10. Vérification finale avant split
    print(f"\nVérification X: NaN = {np.isnan(X.values).sum()}, Inf = {np.isinf(X.values).sum()}")
    print(f"Vérification y: NaN = {np.isnan(y).sum()}")
    
    # 11. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 12. Mise à l'échelle (StandardScaler)
    num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()
    
    # Créer des copies pour éviter les problèmes
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    # Scaler les colonnes numériques
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    # 13. Vérification finale après scaling
    print(f"\nAprès scaling:")
    print(f"X_train: NaN = {np.isnan(X_train.values).sum()}, Inf = {np.isinf(X_train.values).sum()}")
    print(f"X_test: NaN = {np.isnan(X_test.values).sum()}, Inf = {np.isinf(X_test.values).sum()}")
    
    # 14. Sauvegarde du Scaler
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, SCALER_PATH)
    
    # 15. Sauvegarde des données traitées
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)
    
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)
    
    print("\n✅ Prétraitement terminé. Données prêtes pour l'entraînement.")
    print(f"Shapes finales: X_train {X_train.shape}, X_test {X_test.shape}")

if __name__ == '__main__':
    run_preprocessing()