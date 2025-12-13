# Projet MLOps : Prédiction du Customer Churn Télécoms

## 1. Objectif du Projet

Ce projet a pour but de construire un pipeline MLOps complet (de l'entraînement au déploiement) pour prédire la résiliation des clients (**Customer Churn**) dans le secteur des télécommunications. L'objectif est de déployer un modèle de **Classification Binaire** via une API pour une utilisation en production.

---

## 2. Structure du Dépôt

Notre dépôt suit une structure standard de projet Data Science/ML pour garantir la modularité et la séparation des responsabilités.
```
Telco_Churn/
├── .dvc/                    # Configuration DVC
├── .github/                 # Workflows CI/CD (GitHub Actions)
├── .pytest_cache/           # Cache des tests pytest
├── data/                    # Données du projet (versionné par DVC)
│   ├── raw/                 # Données brutes
│   └── processed/           # Données nettoyées/transformées
├── mlruns/                  # Logs et artefacts MLflow
├── models/                  # Modèles sérialisés (.pkl, etc.)
├── monitoring_logs/         # Logs de monitoring en production
├── notebooks/               # Notebooks Jupyter pour EDA et expérimentation
│   ├── 01_EDA.ipynb
│   └── 02_Experimentation.ipynb
├── src/                     # Code source de production
│   ├── data.py              # Prétraitement des données
│   ├── model.py             # Entraînement et évaluation du modèle
│   └── api.py               # API FastAPI pour les prédictions
├── tests/                   # Tests unitaires et d'intégration
├── venv/                    # Environnement virtuel Python
├── .dockerignore            # Fichiers à exclure du build Docker
├── .dvcignore               # Fichiers à exclure du versioning DVC
├── .gitignore               # Fichiers à exclure du versioning Git
├── deploy.sh                # Script de déploiement
├── deploy_local.sh          # Script de déploiement local
├── Dockerfile               # Configuration Docker
├── dvc.lock                 # Fichier de lock DVC
├── dvc.yaml                 # Pipeline DVC
├── mlflow.db                # Base de données MLflow SQLite
├── README.md                # Documentation du projet
└── requirements.txt         # Dépendances Python
```

---

## 3. Technologies MLOps

Ce projet s'appuie sur les outils suivants pour garantir la reproductibilité et l'automatisation :

| Outil | Usage |
|:------|:------|
| **Git** | Versioning du code source |
| **DVC** | Versioning des données et des modèles |
| **MLflow** | Tracking des expériences et gestion des modèles |
| **FastAPI** | API REST pour les prédictions |
| **Docker** | Containerisation de l'application |
| **GitHub Actions** | CI/CD et automatisation des tests |
| **pytest** | Tests unitaires et d'intégration |

---

## 4. Installation et Lancement

### 4.1 Prérequis

- Python 3.8+
- Git
- Docker (optionnel, pour le déploiement containerisé)

### 4.2 Configuration de l'environnement

1. **Cloner le dépôt :**
```bash
   git clone <URL_DE_VOTRE_DÉPÔT>
   cd Telco_Churn
```

2. **Créer et activer l'environnement virtuel :**
```bash
   python -m venv venv
   
   # Sous Windows PowerShell
   .\venv\Scripts\activate
   
   # Sous Linux/macOS
   # source venv/bin/activate
```

3. **Installer les dépendances :**
```bash
   pip install -r requirements.txt
```

4. **Récupérer les données via DVC :**
```bash
   dvc pull
```

### 4.3 Entraînement du modèle
```bash
# Exécuter le pipeline DVC complet
dvc repro

# Ou entraîner manuellement
python src/model.py
```

### 4.4 Lancer l'API en local
```bash
# Avec le script de déploiement local
bash deploy_local.sh

# Ou manuellement avec uvicorn
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

L'API sera accessible sur `http://localhost:8000`

### 4.5 Déploiement avec Docker
```bash
# Construire l'image Docker
docker build -t telco-churn-api .

# Lancer le conteneur
docker run -p 8000:8000 telco-churn-api

# Ou utiliser le script de déploiement
bash deploy.sh
```

---

## 5. Tests

Exécuter les tests unitaires et d'intégration :
```bash
pytest tests/ -v
```

---

## 6. MLflow Tracking

Pour visualiser les expériences et les métriques :
```bash
mlflow ui
```

Accédez à l'interface MLflow sur `http://localhost:5000`

---

## 7. Utilisation de l'API

### Endpoint de prédiction

**POST** `/predict`

**Exemple de requête :**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "feature1": value1,
       "feature2": value2,
       ...
     }'
```

**Réponse :**
```json
{
  "churn_prediction": 0,
  "churn_probability": 0.23
}
```

### Documentation interactive

Une fois l'API lancée, accédez à la documentation Swagger :
- **Swagger UI :** `http://localhost:8000/docs`
- **ReDoc :** `http://localhost:8000/redoc`

---

## 8. Pipeline CI/CD

Le projet utilise GitHub Actions pour l'automatisation :

- **Tests automatiques** à chaque push
- **Linting et vérification du code**
- **Build Docker** automatique
- **Déploiement** (selon configuration)

Les workflows sont définis dans `.github/workflows/`

---

## 9. Monitoring

Les logs de monitoring sont stockés dans `monitoring_logs/` pour suivre :
- Les performances du modèle en production
- Les dérives de données (data drift)
- Les métriques d'utilisation de l'API

---

## 10. Contributeurs

- **Votre Nom** - Développeur Principal

---

## 11. Licence

Ce projet est sous licence [MIT/Apache/autre] - voir le fichier LICENSE pour plus de détails.

---

## 12. Ressources Utiles

- [Documentation DVC](https://dvc.org/doc)
- [Documentation MLflow](https://mlflow.org/docs/latest/index.html)
- [Documentation FastAPI](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)