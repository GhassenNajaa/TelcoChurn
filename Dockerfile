FROM python:3.11-slim

# Mise à jour du système et installation des outils nécessaires (git pour DVC)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

# Définit le répertoire de travail principal du conteneur
WORKDIR /app

# 1. Installation des dépendances
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

# Assure l'installation de DVC et Joblib (si non inclus dans requirements.txt)
RUN pip install dvc joblib

# Copie de tout le projet (code, liens DVC, métadonnées de configuration)
COPY . . 

# --- ÉTAPES CRITIQUES POUR DVC (Chargement du Modèle) ---

# 2. Copie du cache local DVC (contient les fichiers réels .pkl)
# Cette étape nécessite que le dossier .dvc/cache ne soit PAS ignoré dans .dockerignore.
COPY .dvc/cache /app/.dvc/cache

# 3. Configure DVC pour utiliser ce cache interne copié comme source
RUN dvc config cache.dir /app/.dvc/cache

# 4. Exécute 'dvc pull' pour remplir le dossier models/ avec les fichiers réels
RUN dvc pull

# --- EXÉCUTION ---

# Expose le port d'écoute de l'application
EXPOSE 8080

# Commande pour lancer l'API en production avec Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api:app"]