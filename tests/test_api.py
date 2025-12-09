import pytest
import json
from src.api import app

@pytest.fixture
def client():
    """
    Fixture pour créer un client de test Flask.
    """
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """
    Test du endpoint /health
    """
    response = client.get('/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'model_loaded' in data


def test_metrics_endpoint(client):
    """
    Test du endpoint /metrics
    """
    response = client.get('/metrics')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'total_predictions' in data


def test_predict_endpoint_valid_input(client):
    """
    Test du endpoint /predict avec des données valides
    """
    payload = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.5,
        "TotalCharges": 600.0
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'prediction' in data
    assert 'probability_churn' in data
    assert data['prediction'] in [0, 1]
    assert 0 <= data['probability_churn'] <= 1


def test_predict_endpoint_missing_fields(client):
    """
    Test du endpoint /predict avec des données incomplètes
    """
    payload = {
        "gender": "Male",
        "tenure": 12
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    
    # Devrait retourner une erreur 400
    assert response.status_code == 400
    
    data = json.loads(response.data)
    assert 'error' in data


def test_predict_endpoint_invalid_json(client):
    """
    Test du endpoint /predict avec un JSON invalide
    """
    response = client.post(
        '/predict',
        data='invalid json',
        content_type='application/json'
    )
    
    assert response.status_code == 400


def test_response_time(client):
    """
    Test que le temps de réponse est acceptable
    """
    import time
    
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85
    }
    
    start = time.time()
    response = client.post(
        '/predict',
        data=json.dumps(payload),
        content_type='application/json'
    )
    duration = time.time() - start
    
    assert response.status_code == 200
    # Le temps de réponse devrait être < 2 secondes
    assert duration < 2.0, f"Response took {duration:.2f}s, should be < 2.0s"