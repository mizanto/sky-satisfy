import pytest
from backend.prediction_service import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_with_valid_data(client):
    valid_data = {
        "age": 36,
        "class": "business",
        "customer_type": "loyal_customer",
        "ease_of_online_booking": 5,
        "flight_distance": 2000,
        "online_boarding": 5,
        "type_of_travel": "business_travel"
    }
    response = client.post('/predict', json=valid_data)
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}"
    )
    assert 'prediction' in response.json, "'prediction' key not in response"


def test_predict_with_invalid_data(client):
    invalid_data = {'invalid': 'data'}
    response = client.post('/predict', json=invalid_data)
    assert response.status_code == 400, (
        f"Expected status code 400, got {response.status_code}"
    )
    assert 'error' in response.json, "'error' key not in response"


def test_predict_with_missing_fields(client):
    missing_fields_data = {'customer_type': 'loyal_customer', 'age': 25}
    response = client.post('/predict', json=missing_fields_data)
    assert response.status_code == 400, (
        f"Expected status code 400, got {response.status_code}"
    )
    assert 'error' in response.json, "'error' key not in response"


def test_model_info(client):
    response = client.get('/model/info')
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}"
    )
    json_data = response.get_json()
    assert 'model_type' in json_data, "'model_type' key not in response"
    assert 'training_date' in json_data, "'training_date' key not in response"
    assert 'metrics' in json_data, "'metrics' key not in response"
    assert isinstance(json_data['model_type'], str), (
        f"Expected str, got {type(json_data['model_type'])}"
    )
    assert isinstance(json_data['training_date'], str), (
        f"Expected str, got {type(json_data['training_date'])}"
    )
    assert isinstance(json_data['metrics'], dict), (
        f"Expected dict, got {type(json_data['metrics'])}"
    )


def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200, (
        f"Expected status code 200, got {response.status_code}"
    )
    assert response.get_json() == {'status': 'OK'}, "Expected {'status': 'OK'}"
