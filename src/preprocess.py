# src/preprocess.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib # Utilisé pour sauvegarder le scaler

# --- Constantes MLOps ---
RAW_DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
PROCESSED_DATA_DIR = 'data/processed'
SCALER_PATH = 'models/scaler.pkl' 
TEST_SIZE = 0.3
RANDOM_STATE = 40

def run_preprocessing():
    """Charge, nettoie, transforme, scale, split et sauvegarde les données."""

    # 1. Chargement et Nettoyage (Basé sur votre Notebook)
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(['customerID'], axis = 1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Gérer les NaNs (basé sur votre code)
    # Supprime les lignes avec tenure 0
    df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
    # Remplit les NaNs restants dans TotalCharges avec la moyenne
    df.fillna(df["TotalCharges"].mean(), inplace=True) 

    # Transformation de SeniorCitizen en int
    df["SeniorCitizen"]= df["SeniorCitizen"].map({"No": 0, "Yes": 1})

    # 2. Encodage (Adaptation de votre LabelEncoder générique)
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # 3. Séparation X/y
    X = df.drop(columns = ['Churn'])
    y = df['Churn'].values

    # 4. Split Train/Test (Stratification pour l'équilibre des classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    # 5. Mise à l'échelle (StandardScaler)
    num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    scaler = StandardScaler()

    # FIT SEULEMENT sur X_train, puis TRANSFORM sur les deux ensembles
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 6. Sauvegarde du Scaler MLOps (Pour le déploiement!)
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(scaler, SCALER_PATH) 

    # 7. Sauvegarde des données traitées
    if not os.path.exists(PROCESSED_DATA_DIR): os.makedirs(PROCESSED_DATA_DIR)
    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print("Prétraitement terminé. Données prêtes pour l'entraînement.")

if __name__ == '__main__':
    run_preprocessing()