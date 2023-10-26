import pandas as pd
from src.utils.data_loader import load_data, preprocess_data


def test_load_data():
    df = load_data('tests/fakes/fake_data.csv')
    assert isinstance(df, pd.DataFrame)


def test_preprocess_data():
    df = pd.DataFrame({
        'satisfaction': [1, 0],
        'Customer Type': ['Loyal', 'Disloyal'],
        'Age': [25, 30],
        'Type of Travel': ['Business', 'Personal'],
        'Class': ['Eco', 'Business'],
        'Flight Distance': [400, 200],
        'Ease of Online booking': [3, 4],
        'Online boarding': [4, 3]
    })
    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'satisfaction' not in X.columns
    assert y.equals(df['satisfaction'])
