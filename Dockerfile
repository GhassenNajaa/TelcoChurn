FROM python:3.11-slim

# Mise à jour du système
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Répertoire de travail
WORKDIR /app

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copier le code de l'application
COPY src/ ./src/
COPY models/ ./models/
COPY assets/ ./assets/   # si tu as un dossier assets
COPY app.py .
COPY gunicorn_conf.py .  # si tu utilises une config gunicorn (optionnel)

# Exposer le port API
EXPOSE 8080

# Lancement de l'API via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "src.api:app"]
