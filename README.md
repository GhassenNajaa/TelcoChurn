# Projet MLOps : Prédiction du Customer Churn Télécoms

## 1. Objectif du Projet

Ce projet a pour but de construire un pipeline MLOps complet (de l'entraînement au déploiement) pour prédire la résiliation des clients (**Customer Churn**) dans le secteur des télécommunications. L'objectif est de déployer un modèle de **Classification Binaire** via une API pour une utilisation en production.

---

## ⚙️ 2. Structure du Dépôt

Notre dépôt suit une structure standard de projet Data Science/ML pour garantir la modularité et la séparation des responsabilités.

| Dossier | Contenu | Rôle |
| :--- | :--- | :--- |
| **`data/`** | `raw/`, `processed/` | Données brutes et nettoyées. **Versionné par DVC.** |
| **`src/`** | `data.py`, `model.py`, `api.py` | Le code de production : preprocessing, entraînement, et API. |
| **`notebooks/`** | `01_EDA.ipynb`, `02_Experimentation.ipynb` | Analyse exploratoire et développement des prototypes. |
| **`models/`** | Fichiers de modèles sérialisés (`.pkl`, etc.) | Stockage des modèles avant déploiement. |
| **`tests/`** | `test_...py` | Code pour les tests unitaires et d'intégration. |
| **`config/`** | Fichiers de configuration (ex: YAML) | Paramètres du pipeline et hyperparamètres du modèle. |

---

##  3. Technologies MLOps

Ce projet s'appuie sur les outils suivants pour garantir la reproductibilité :

* **Versioning du Code :** **Git**
* **Versioning des Données :** **DVC** (Data Version Control)
* **Tracking d'Expérience :** **MLflow**
* **Containerisation & Déploiement :** **Docker** & **FastAPI**
* **Automatisation CI/CD :** **GitHub Actions**

---

## 🚀 4. Installation et Lancement

Pour cloner et configurer l'environnement de développement pour la première fois :

1.  **Cloner le dépôt :**
    ```bash
    git clone <URL_DE_VOTRE_DÉPÔT>
    cd Projet_MLOps
    ```
2.  **Créer et activer l'environnement virtuel :**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # Sous Windows PowerShell
    # source venv/bin/activate # Sous Linux/macOS
    ```
3.  **Installer les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Récupérer les Données (via DVC) :**
    ```bash
    # Récupère le dataset brut versionné dans data/raw/
    dvc pull
    ```