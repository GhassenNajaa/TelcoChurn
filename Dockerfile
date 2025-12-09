FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    openssh-client && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Mettre à jour pip
RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout 1000

RUN pip install dvc joblib

COPY . . 

RUN mkdir -p data/processed models mlruns

# EXPOSE le port où l'API écoute (optionnel mais bonne pratique)
EXPOSE 8080

# Ceci exécute src/api.py en production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api:app"]