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

# CORRECTION CLÉ : Ajout de --no-scm pour éviter l'erreur Git
CMD ["dvc", "repro", "--no-commit"]