# src/train.py

import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import joblib

# Importations des modèles et des métriques
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# --- Constantes MLOps ---
PROCESSED_DATA_DIR = 'data/processed'
RANDOM_STATE = 40 
MLFLOW_EXPERIMENT_NAME = "Telco_Churn_Final_Model_Selection"
CHAMPION_MODEL_PATH = 'models/model_champion.pkl'

# --- Fonctions MLOps de Base ---

def load_processed_data():
    """Charge les ensembles de données prétraités."""
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        y_test = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_test.npy'))
        return X_train, X_test, y_train, y_test
    except FileNotFoundError:
        print("ERREUR : Les données prétraitées sont introuvables. Avez-vous exécuté src/preprocess.py ?")
        exit()
        
def track_model_performance(model, params, X_train, X_test, y_train, y_test, run_name):
    """Fonction générique pour entraîner, évaluer et tracker un modèle avec MLflow."""

    model_type = type(model).__name__
    
    with mlflow.start_run(run_name=run_name) as run:
        
        print(f"\nEntraînement: {run_name} ({model_type})...")
        
        # 1. Entraînement
        model.set_params(**params)
        model.fit(X_train, y_train)

        # 2. Prédiction et Évaluation
        y_pred = model.predict(X_test)
        
        auc_score = None
        if hasattr(model, "predict_proba"):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
            except Exception:
                pass 
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        
        if auc_score is not None:
            metrics["roc_auc"] = auc_score

        # 3. Tracking des informations
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model_artifact")
        
        # Affichage des résultats
        print(f"   Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}")
        if auc_score:
            print(f"   ROC-AUC: {auc_score:.4f}")
        
        return run.info.run_id, metrics.get("roc_auc", 0), metrics

# --- Fonction d'Optimisation du Modèle Champion ---

def find_champion_model(X_train, X_test, y_train, y_test):
    """Effectue l'exploration, la comparaison, puis l'optimisation par GridSearch."""
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    print("\n" + "="*70)
    print("PHASE 1 : EXPLORATION DES MODELES BASELINE")
    print("="*70)
    
    # --- Étape 1: Définition des Modèles d'Exploration ---
    
    models_to_test = [
        (KNeighborsClassifier(n_jobs=-1), {"n_neighbors": 11}, "KNN_Exploration"), 
        (SVC(random_state=RANDOM_STATE, probability=True), {"kernel": "rbf", "C": 1.0}, "SVC_Exploration"), 
        (DecisionTreeClassifier(random_state=RANDOM_STATE), {"max_depth": 5}, "DecisionTree_Exploration"),
        (LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), {"C": 0.5, "solver": "liblinear"}, "LogisticRegression_Exploration"), 
        (RandomForestClassifier(random_state=RANDOM_STATE), {"n_estimators": 500, "max_depth": 10}, "RandomForest_Exploration"), 
        (AdaBoostClassifier(random_state=RANDOM_STATE), {"n_estimators": 100, "learning_rate": 0.1}, "AdaBoost_Exploration"),
        (GradientBoostingClassifier(random_state=RANDOM_STATE), {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}, "GradientBoosting_Exploration")
    ]
    
    all_runs = []
    
    # Exécuter et tracker les modèles d'exploration
    for model, params, name in models_to_test:
        run_id, auc, metrics = track_model_performance(model, params, X_train, X_test, y_train, y_test, name)
        all_runs.append({'name': name, 'model': model, 'auc': auc, 'metrics': metrics})

    # Le Voting Classifier
    print("\nTest du VotingClassifier (Ensemble)...")
    clf1 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
    clf2 = LogisticRegression(C=0.5, solver='liblinear', random_state=RANDOM_STATE, max_iter=1000)
    clf3 = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=RANDOM_STATE)
    voting_model = VotingClassifier(estimators=[('gbc', clf1), ('lr', clf2), ('abc', clf3)], voting='soft')
    
    run_id, auc, metrics = track_model_performance(voting_model, {"voting": "soft"}, X_train, X_test, y_train, y_test, "VotingClassifier_Exploration")
    all_runs.append({'name': "VotingClassifier_Exploration", 'model': voting_model, 'auc': auc, 'metrics': metrics})

    # --- Étape 2: Identifier le Modèle Champion ---
    
    print("\n" + "="*70)
    print("PHASE 2 : SELECTION DU CHAMPION")
    print("="*70)
    
    champion_run = max(all_runs, key=lambda x: x['auc'])
    print(f"\nModele Champion: {champion_run['name']}")
    print(f"   ROC-AUC: {champion_run['auc']:.4f}")
    print(f"   Accuracy: {champion_run['metrics']['accuracy']:.4f}")
    print(f"   F1-Score: {champion_run['metrics']['f1_score']:.4f}")
    
    # Afficher le classement complet
    print("\nClassement complet (par ROC-AUC):")
    sorted_runs = sorted(all_runs, key=lambda x: x['auc'], reverse=True)
    for i, run in enumerate(sorted_runs, 1):
        marker = "[1]" if i == 1 else "[2]" if i == 2 else "[3]" if i == 3 else "   "
        print(f"   {marker} {i}. {run['name']:<35} | AUC: {run['auc']:.4f} | F1: {run['metrics']['f1_score']:.4f}")
    
    # --- Étape 3: Optimisation du Champion (GridSearchCV) ---

    print("\n" + "="*70)
    print("PHASE 3 : HYPERPARAMETER TUNING (GridSearchCV)")
    print("="*70)

    # Choix intelligent du modèle à optimiser
    # Si le champion est GradientBoosting ou RandomForest, on l'optimise
    # Sinon, on optimise GradientBoosting par défaut (car c'est souvent le meilleur)
    
    champion_model_type = type(champion_run['model']).__name__
    
    if 'GradientBoosting' in champion_run['name']:
        print("\nOptimisation du GradientBoostingClassifier (Champion detecte)")
        final_best_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        tuning_name = "CHAMPION_FINAL_TUNED_GB"
        
    elif 'RandomForest' in champion_run['name']:
        print("\nOptimisation du RandomForestClassifier (Champion detecte)")
        final_best_model = RandomForestClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [200, 500, 700],
            'max_depth': [10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        tuning_name = "CHAMPION_FINAL_TUNED_RF"
        
    else:
        print(f"\nChampion detecte: {champion_model_type}")
        print("   Optimisation de GradientBoosting par defaut (modele robuste)")
        final_best_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        tuning_name = "CHAMPION_FINAL_TUNED_GB"
    
    print(f"\nGrille de recherche: {len(param_grid)} hyperparametres")
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"   Nombre total de combinaisons: {total_combinations}")
    print(f"   Cross-validation: 3-fold")
    print(f"   Metrique d'optimisation: ROC-AUC")
    
    grid_search = GridSearchCV(
        estimator=final_best_model, 
        param_grid=param_grid, 
        scoring='roc_auc', 
        cv=3, 
        n_jobs=-1, 
        verbose=1,  # Afficher la progression
        return_train_score=True
    )
    
    print("\nRecherche en cours (cela peut prendre quelques minutes)...\n")
    grid_search.fit(X_train, y_train)
    
    # --- Étape 4: Tracking Final du Modèle Optimisé ---

    tuned_champion_model = grid_search.best_estimator_
    tuned_best_params = grid_search.best_params_
    
    print("\n" + "="*70)
    print("RESULTATS DE L'OPTIMISATION (GridSearchCV)")
    print("="*70)
    print(f"\nMeilleurs hyperparametres trouves:")
    for param, value in tuned_best_params.items():
        print(f"   {param}: {value}")
    
    print(f"\nMeilleur score CV (ROC-AUC): {grid_search.best_score_:.4f}")
    
    # Track de la run finale dans MLflow
    run_id, tuned_auc, tuned_metrics = track_model_performance(
        tuned_champion_model, 
        tuned_best_params, 
        X_train, X_test, y_train, y_test, 
        tuning_name
    )
    
    # CRITIQUE: Ajouter le modèle tuné à la liste pour comparaison globale
    all_runs.append({
        'name': tuning_name, 
        'model': tuned_champion_model, 
        'auc': tuned_auc, 
        'metrics': tuned_metrics
    })
    
    # --- ÉTAPE CRITIQUE: Sélection du CHAMPION ABSOLU parmi TOUS les modèles ---
    print("\n" + "="*70)
    print("SELECTION DU CHAMPION ABSOLU (Tous modeles confondus)")
    print("="*70)
    
    # Re-trier tous les modèles (baseline + tuned)
    final_champion_run = max(all_runs, key=lambda x: x['auc'])
    
    print("\nTop 5 des meilleurs modeles (ROC-AUC):")
    sorted_all_runs = sorted(all_runs, key=lambda x: x['auc'], reverse=True)
    for i, run in enumerate(sorted_all_runs[:5], 1):
        marker = "[1]" if i == 1 else "[2]" if i == 2 else "[3]" if i == 3 else "   "
        print(f"   {marker} {i}. {run['name']:<40} | AUC: {run['auc']:.4f} | F1: {run['metrics']['f1_score']:.4f}")
    
    # Afficher les détails du champion absolu
    print(f"\nCHAMPION ABSOLU: {final_champion_run['name']}")
    print(f"   ROC-AUC:   {final_champion_run['auc']:.4f}")
    print(f"   Accuracy:  {final_champion_run['metrics']['accuracy']:.4f}")
    print(f"   F1-Score:  {final_champion_run['metrics']['f1_score']:.4f}")
    print(f"   Precision: {final_champion_run['metrics']['precision']:.4f}")
    print(f"   Recall:    {final_champion_run['metrics']['recall']:.4f}")
    
    # Comparaison avec le champion d'exploration initial
    if final_champion_run['name'] == tuning_name:
        improvement = ((tuned_auc - champion_run['auc']) / champion_run['auc']) * 100
        print(f"\nLe GridSearch a ameliore les performances!")
        print(f"   Amelioration vs champion baseline: {improvement:+.2f}%")
    elif final_champion_run['name'] == champion_run['name']:
        print(f"\nLe champion d'exploration reste le meilleur")
        print(f"   Le GridSearch n'a pas apporte d'amelioration")
    else:
        print(f"\nUn autre modele s'est revele etre le meilleur!")
    
    # Assigner le champion final
    final_champion_model = final_champion_run['model']
    final_champion_name = final_champion_run['name']
    final_auc = final_champion_run['auc']
    final_metrics = final_champion_run['metrics']
    
    # Sauvegarde du VRAI champion final sur disque
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(final_champion_model, CHAMPION_MODEL_PATH)
    print(f"\nChampion sauvegarde: {CHAMPION_MODEL_PATH}")
    
    # Log du modèle final en artefact MLflow avec run dédiée
    with mlflow.start_run(run_name=f"FINAL_CHAMPION_{final_champion_name}") as final_run:
        mlflow.log_param("champion_model", final_champion_name)
        mlflow.log_param("selection_method", "Best ROC-AUC on Test Set")
        mlflow.log_metrics({
            "champion_roc_auc": final_auc,
            "champion_accuracy": final_metrics['accuracy'],
            "champion_f1_score": final_metrics['f1_score'],
            "champion_precision": final_metrics['precision'],
            "champion_recall": final_metrics['recall']
        })
        mlflow.sklearn.log_model(final_champion_model, "champion_model")
        mlflow.log_artifact(CHAMPION_MODEL_PATH, "final_model_deployment")
    
    # Affichage du rapport de classification final
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT - CHAMPION FINAL")
    print("="*70)
    y_pred_final = final_champion_model.predict(X_test)
    print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))
    
    print("\n" + "="*70)
    print("PROCESSUS TERMINE")
    print("="*70)
    print(f"Champion: {final_champion_name}")
    print(f"ROC-AUC: {final_auc:.4f}")
    print(f"Fichier: {CHAMPION_MODEL_PATH}")
    print(f"Experience MLflow: {MLFLOW_EXPERIMENT_NAME}")
    print(f"Lancez 'mlflow ui' pour voir tous les resultats")
    print("="*70 + "\n")


if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
        
    X_train, X_test, y_train, y_test = load_processed_data()
    find_champion_model(X_train, X_test, y_train, y_test)