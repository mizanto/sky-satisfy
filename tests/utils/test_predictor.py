from src.utils.predictor import make_prediction
from unittest.mock import patch


@patch('src.utils.model_serializer.load_model')
def test_make_prediction(mock_load_model):
    mock_model = 'fake_model'
    mock_load_model.return_value = mock_model

    # Test data
    test_data = {
        'customer_type': 1,
        'age': 35,
        'type_of_travel': 1,
        'flight_distance': 500,
        'ease_of_online_booking': 3,
        'online_boarding': 4,
        'class_business': 1,
        'class_eco': 0,
        'class_eco_plus': 0
    }

    # Make prediction
    prediction = make_prediction(mock_model, test_data)

    assert prediction is not None
    assert isinstance(prediction, (dict, list))

    if isinstance(prediction, dict):
        assert 'error' in prediction
    else:
        assert prediction[0] in [0, 1]
