#!/bin/bash

# Fichier: deploy_local.sh
# Description: Script pour tirer (pull) la dernière image Docker depuis Docker Hub 
# et redémarrer le conteneur API sur la machine locale.

# --- Configuration ---
# Remplacez 177777771 par votre Nom d'utilisateur Docker Hub
IMAGE_NAME="177777771/telco-churn-api:latest" 
CONTAINER_NAME="telco-api-prod"
LOCAL_PORT="8081"
CONTAINER_PORT="8080"

# --- Déploiement ---

echo "--- Déploiement Manuel Local de l'API Telco Churn ---"

# 1. Tirer la dernière image depuis Docker Hub
echo "[1/4] Tirage de la dernière image ($IMAGE_NAME)..."
docker pull $IMAGE_NAME

# 2. Arrêter l'ancien conteneur s'il est en cours d'exécution
echo "[2/4] Arrêt de l'ancien conteneur ($CONTAINER_NAME)..."
# La commande '|| true' garantit que le script ne s'arrête pas si le conteneur n'existe pas encore.
docker stop $CONTAINER_NAME 2> /dev/null || true

# 3. Supprimer l'ancien conteneur
echo "[3/4] Suppression de l'ancien conteneur..."
docker rm $CONTAINER_NAME 2> /dev/null || true

# 4. Lancer un nouveau conteneur avec la dernière image tirée
echo "[4/4] Lancement du nouveau conteneur sur le port $LOCAL_PORT..."
docker run -d \
  --name $CONTAINER_NAME \
  -p $LOCAL_PORT:$CONTAINER_PORT \
  $IMAGE_NAME

echo "--- Déploiement Terminé ---"
echo "API accessible sur : http://localhost:$LOCAL_PORT/predict"