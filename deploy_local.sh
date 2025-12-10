#!/bin/bash

# Fichier: deploy_local.sh
# Description: Script pour pull la dernière image Docker Hub et redémarrer le conteneur

# --- Configuration ---
IMAGE_NAME="177777771/telco-churn-api:latest" 
CONTAINER_NAME="telco-api-prod"
LOCAL_PORT="8081"
CONTAINER_PORT="8080"

# --- Déploiement ---

echo "=========================================="
echo "  Déploiement Local - Telco Churn API"
echo "=========================================="

# 1. Pull de la dernière image
echo ""
echo "[1/4]  Pull de la dernière image depuis Docker Hub..."
if docker pull $IMAGE_NAME; then
    echo "Image récupérée avec succès"
else
    echo "Erreur lors du pull de l'image"
    exit 1
fi

# 2. Arrêter l'ancien conteneur
echo ""
echo "[2/4] Arrêt de l'ancien conteneur..."
if docker stop $CONTAINER_NAME 2>/dev/null; then
    echo "Conteneur arrêté"
else
    echo "Aucun conteneur à arrêter"
fi

# 3. Supprimer l'ancien conteneur
echo ""
echo "[3/4] Suppression de l'ancien conteneur..."
if docker rm $CONTAINER_NAME 2>/dev/null; then
    echo " Conteneur supprimé"
else
    echo " Aucun conteneur à supprimer"
fi

# 4. Lancer le nouveau conteneur
echo ""
echo "[4/4] Lancement du nouveau conteneur..."
if docker run -d \
  --name $CONTAINER_NAME \
  -p $LOCAL_PORT:$CONTAINER_PORT \
  --restart unless-stopped \
  $IMAGE_NAME; then
    echo "Conteneur démarré avec succès"
else
    echo "Erreur lors du lancement du conteneur"
    exit 1
fi

# Vérification finale
echo ""
echo "=========================================="
echo "Déploiement terminé !"
echo "=========================================="
echo ""
echo " API accessible sur:"
echo "   http://localhost:$LOCAL_PORT/predict"
echo ""
echo "Vérifier les logs:"
echo "   docker logs -f $CONTAINER_NAME"
echo ""
echo " Tester l'API:"
echo "   curl -X POST http://localhost:$LOCAL_PORT/predict -H 'Content-Type: application/json' -d '{...}'"
echo ""