#!/bin/bash

# ========================================
# deploy.sh - Script universel de déploiement
# Usage: ./deploy.sh [dev|staging|prod]
# ========================================

ENVIRONMENT=${1:-prod}  # Par défaut : prod

# Configuration selon l'environnement
case $ENVIRONMENT in
  dev)
    IMAGE_TAG="dev"
    CONTAINER_NAME="telco-api-dev"
    LOCAL_PORT="8082"
    echo "🔧 Déploiement en DÉVELOPPEMENT"
    ;;
  staging)
    IMAGE_TAG="staging"
    CONTAINER_NAME="telco-api-staging"
    LOCAL_PORT="8081"
    echo "🧪 Déploiement en STAGING"
    ;;
  prod)
    IMAGE_TAG="latest"
    CONTAINER_NAME="telco-api-prod"
    LOCAL_PORT="8080"
    echo " Déploiement en PRODUCTION"
    ;;
  *)
    echo " Environnement invalide : $ENVIRONMENT"
    echo "Usage: ./deploy.sh [dev|staging|prod]"
    exit 1
    ;;
esac

IMAGE_NAME="177777771/telco-churn-api:$IMAGE_TAG"
CONTAINER_PORT="8080"

echo "=========================================="
echo "  Déploiement Telco Churn API"
echo "  Environnement : $ENVIRONMENT"
echo "=========================================="

# 1. Pull de l'image
echo ""
echo "[1/4]  Pull de l'image depuis Docker Hub..."
if docker pull $IMAGE_NAME; then
    echo " Image récupérée : $IMAGE_NAME"
else
    echo " Erreur lors du pull de l'image"
    exit 1
fi

# 2. Arrêter l'ancien conteneur
echo ""
echo "[2/4]  Arrêt de l'ancien conteneur..."
if docker stop $CONTAINER_NAME 2>/dev/null; then
    echo " Conteneur arrêté"
else
    echo " Aucun conteneur à arrêter"
fi

# 3. Supprimer l'ancien conteneur
echo ""
echo "[3/4]   Suppression de l'ancien conteneur..."
if docker rm $CONTAINER_NAME 2>/dev/null; then
    echo " Conteneur supprimé"
else
    echo "  Aucun conteneur à supprimer"
fi

# 4. Lancer le nouveau conteneur
echo ""
echo "[4/4] 🚀 Lancement du nouveau conteneur..."
if docker run -d \
  --name $CONTAINER_NAME \
  -p $LOCAL_PORT:$CONTAINER_PORT \
  --restart unless-stopped \
  -e ENVIRONMENT=$ENVIRONMENT \
  $IMAGE_NAME; then
    echo " Conteneur démarré avec succès"
else
    echo " Erreur lors du lancement du conteneur"
    exit 1
fi

# Vérification finale
echo ""
echo "=========================================="
echo "Déploiement terminé !"
echo "=========================================="
echo ""
echo "📊 Informations :"
echo "   Environnement : $ENVIRONMENT"
echo "   Conteneur     : $CONTAINER_NAME"
echo "   Port local    : $LOCAL_PORT"
echo "   Image         : $IMAGE_NAME"
echo ""
echo "API accessible sur :"
echo "   http://localhost:$LOCAL_PORT/health"
echo "   http://localhost:$LOCAL_PORT/predict"
echo "   http://localhost:$LOCAL_PORT/metrics"
echo ""
echo "Commandes utiles :"
echo "   Logs     : docker logs -f $CONTAINER_NAME"
echo "   Status   : docker ps | grep $CONTAINER_NAME"
echo "   Stop     : docker stop $CONTAINER_NAME"
echo "   Restart  : docker restart $CONTAINER_NAME"
echo ""
echo "Test rapide :"
echo "   curl http://localhost:$LOCAL_PORT/health"
echo ""
echo "=========================================="