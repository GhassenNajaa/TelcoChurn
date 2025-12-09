import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib

# Importations des modèles
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, classification_report
)

# --- Constantes ---
PROCESSED_DATA_DIR = 'data/processed'
RANDOM_STATE = 40
MLFLOW_EXPERIMENT_NAME = "Telco_Churn_Final_Model_Selection"
CHAMPION_MODEL_PATH = 'models/model_champion.pkl'
FEATURES_PATH = 'models/feature_columns.pkl'


# ------------------------------------------
#     Chargement des données prétraitées
# ------------------------------------------
def load_processed_data():
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
        X_test  = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        y_test  = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("ERREUR : Les données prétraitées sont introuvables. "
              "Exécute d’abord : python src/preprocess.py")
        exit()


# ------------------------------------------
#   Fonction de suivi MLflow d’un modèle
# ------------------------------------------
def track_model_performance(model, params, X_train, X_test, y_train, y_test, run_name):

    with mlflow.start_run(run_name=run_name):
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # AUC si possible
        auc_score = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_proba)
            except:
                pass

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        if auc_score is not None:
            metrics["roc_auc"] = auc_score

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model_artifact")

        return auc_score if auc_score else 0, metrics


# ------------------------------------------
#     FONCTION PRINCIPALE : champion
# ------------------------------------------
def find_champion_model(X_train, X_test, y_train, y_test):

    print("\n==================== PHASE 1 : TEST DES MODELES ====================\n")

    models_to_test = [
        (KNeighborsClassifier(), {"n_neighbors": 11}, "KNN"),
        (SVC(probability=True, random_state=RANDOM_STATE), {"C": 1.0}, "SVC"),
        (DecisionTreeClassifier(random_state=RANDOM_STATE), {"max_depth": 5}, "DecisionTree"),
        (LogisticRegression(max_iter=1000, solver="liblinear"), {"C": 0.5}, "LogReg"),
        (RandomForestClassifier(random_state=RANDOM_STATE), {"n_estimators": 300}, "RandomForest"),
        (AdaBoostClassifier(random_state=RANDOM_STATE), {"n_estimators": 100}, "AdaBoost"),
        (GradientBoostingClassifier(random_state=RANDOM_STATE),
         {"n_estimators": 100}, "GradientBoosting"),
    ]

    all_runs = []

    for model, params, name in models_to_test:
        auc, metrics = track_model_performance(model, params, X_train, X_test, y_train, y_test, name)
        all_runs.append((name, model, auc, metrics))

    # Voting Classifier
    print("\nTest VotingClassifier...")
    voting = VotingClassifier(
        estimators=[
            ('gb', GradientBoostingClassifier()),
            ('lr', LogisticRegression(max_iter=1000, solver="liblinear")),
            ('ab', AdaBoostClassifier())
        ],
        voting='soft'
    )
    auc, metrics = track_model_performance(voting, {"voting": "soft"},
                                           X_train, X_test, y_train, y_test,
                                           "Voting")
    all_runs.append(("Voting", voting, auc, metrics))

    # ------------------------------------
    #   Sélection du meilleur modèle
    # ------------------------------------
    print("\n==================== PHASE 2 : SELECTION DU CHAMPION ====================\n")

    champion = max(all_runs, key=lambda x: x[2])
    champ_name, champ_model, champ_auc, champ_metrics = champion

    print(f"Champion initial : {champ_name} | AUC={champ_auc:.4f}")

    # ------------------------------------
    # Hyperparam tuning du champion
    # ------------------------------------
    print("\n==================== PHASE 3 : TUNING ====================\n")

    # Choix du modèle à tuner
    if champ_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }
    else:
        model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        }

    grid = GridSearchCV(model, param_grid, scoring="roc_auc", cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    tuned_model = grid.best_estimator_
    tuned_params = grid.best_params_

    auc, tuned_metrics = track_model_performance(
        tuned_model, tuned_params, X_train, X_test, y_train, y_test,
        "TUNED_CHAMPION"
    )

    print("\nAUC TUNED :", auc)

    # ------------------------------------
    # Sélection finale
    # ------------------------------------
    final_model = tuned_model if auc > champ_auc else champ_model
    final_name  = "TUNED_CHAMPION" if auc > champ_auc else champ_name
    final_auc   = max(auc, champ_auc)

    print("\n==================== CHAMPION FINAL ====================\n")
    print(f"Champion : {final_name} | AUC = {final_auc:.4f}")

    # Sauvegarde du modèle champion
    joblib.dump(final_model, CHAMPION_MODEL_PATH)

    print(f"Modèle champion enregistré sous : {CHAMPION_MODEL_PATH}")

    return final_model


# ------------------------------------------
#   MAIN
# ------------------------------------------
if __name__ == '__main__':

    if not os.path.exists('models'):
        os.makedirs('models')

    X_train, X_test, y_train, y_test = load_processed_data()

    model = find_champion_model(X_train, X_test, y_train, y_test)

    # Vérifier présence du fichier des features
    if os.path.exists(FEATURES_PATH):
        print(f"Feature columns OK: {FEATURES_PATH}")
    else:
        joblib.dump(list(X_train.columns), FEATURES_PATH)
        print(f"Feature columns sauvegardés dans : {FEATURES_PATH}")

    print("\nENTRAÎNEMENT TERMINÉ ✓\n")
