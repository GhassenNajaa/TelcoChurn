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
ENCODERS_PATH = 'models/encoders.pkl'
FEATURES_PATH = 'models/feature_columns.pkl'
TEST_SIZE = 0.3
RANDOM_STATE = 40

def run_preprocessing():
    """Charge, nettoie, transforme, scale, split et sauvegarde les données.
       Crée et sauvegarde également les encoders (mappings) utilisés en production.
    """
    # 1. Chargement
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(['customerID'], axis=1)

    # 2. Gestion de TotalCharges AVANT tout autre traitement
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 3. Suppression des lignes avec tenure 0 (ou éventuellement imput)
    df = df[df['tenure'] != 0].copy()

    # 4. Remplissage des NaNs dans TotalCharges avec la moyenne
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # 5. Transformation de SeniorCitizen si besoin (ex: "Yes"/"No")
    if df["SeniorCitizen"].dtype == 'object':
        df["SeniorCitizen"] = df["SeniorCitizen"].map({"No": 0, "Yes": 1})

    # 6. Préparer encoders pour les colonnes catégorielles
    encoders = {}
    categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c != 'Churn']

    # Encodage stable : on construit des mappings category -> integer
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        encoders[col] = mapping
        # appliquer l'encodage au DataFrame
        df[col] = df[col].map(mapping).astype(int)

    # Encodage explicite de la target 'Churn' (Yes/No -> 1/0)
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].astype(str).map({'No': 0, 'Yes': 1})
        # Si d'autres valeurs possibles, on remplace les NaN résultants par 0 par défaut
        df['Churn'].fillna(0, inplace=True)

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

    # 10. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 11. Mise à l'échelle (StandardScaler) sur colonnes numériques
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    # créer copies
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 12. Vérification finale après scaling
    print(f"\nAprès scaling:")
    print(f"X_train: NaN = {np.isnan(X_train.values).sum()}, Inf = {np.isinf(X_train.values).sum()}")
    print(f"X_test: NaN = {np.isnan(X_test.values).sum()}, Inf = {np.isinf(X_test.values).sum()}")

    # 13. Sauvegarde des artefacts (scaler, encoders, feature columns)
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    # Sauvegarder l'ordre exact des colonnes (utile pour l'API)
    feature_columns = X_train.columns.tolist()
    joblib.dump(feature_columns, FEATURES_PATH)

    # 14. Sauvegarde des données traitées
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'), y_test)

    print("\nPrétraitement terminé. Données prêtes pour l'entraînement.")
    print(f"Shapes finales: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"Artifacts saved: {SCALER_PATH}, {ENCODERS_PATH}, {FEATURES_PATH}")

if __name__ == '__main__':
    run_preprocessing()
