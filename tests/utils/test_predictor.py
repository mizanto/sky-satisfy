import pandas as pd

from src.utils.predictor import make_prediction, _prepare_data
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


def test_prepare_data():
    df = pd.DataFrame({
        'customer_type': ['loyal_customer', 'disloyal_customer'],
        'type_of_travel': ['business_travel', 'personal_travel'],
        'class_': ['business', 'eco']
    })
    prepared_df = _prepare_data(df)
    assert 'customer_type' in prepared_df.columns
    assert 'type_of_travel' in prepared_df.columns
    assert 'class_business' in prepared_df.columns
    assert 'class_eco' in prepared_df.columns
    assert prepared_df['customer_type'].dtype == 'int'
    assert prepared_df['type_of_travel'].dtype == 'int'


def test_make_prediction_exception():
    with patch('src.utils.predictor._prepare_data') as mock_prepare:
        mock_prepare.side_effect = Exception("Test exception")
        prediction = make_prediction('fake_model', {})
        assert 'error' in prediction
        assert prediction['error'] == "Test exception"
