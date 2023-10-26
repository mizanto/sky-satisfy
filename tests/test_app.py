import pytest
from src.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_predict_with_valid_data(client):
    valid_data = {
        'customer_type': 'loyal_customer',
        'age': 25,
        'type_of_travel': 'business_travel',
        'flight_distance': 500,
        'ease_of_online_booking': 3,
        'online_boarding': 4,
        'class': 'business'
    }
    response = client.post('/predict', json=valid_data)
    assert response.status_code == 200
    assert 'prediction' in response.json


def test_predict_with_invalid_data(client):
    invalid_data = {'invalid': 'data'}
    response = client.post('/predict', json=invalid_data)
    assert response.status_code == 400
    assert 'error' in response.json


def test_predict_with_missing_fields(client):
    missing_fields_data = {
        'customer_type': 'loyal_customer',
        'age': 25,
    }
    response = client.post('/predict', json=missing_fields_data)
    assert response.status_code == 400
    assert 'error' in response.json


def test_model_info(client):
    response = client.get('/model/info')
    assert response.status_code == 200
    json_data = response.get_json()

    assert 'model_type' in json_data
    assert 'training_date' in json_data
    assert 'metrics' in json_data

    assert isinstance(json_data['model_type'], str)
    assert isinstance(json_data['training_date'], str)
    assert isinstance(json_data['metrics'], dict)


def test_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json() == {'status': 'OK'}


def test_api_info(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.content_type == 'text/html; charset=utf-8'
    assert b'API Endpoints' in response.data
