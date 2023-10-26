import pandas as pd
from src.utils.data_loader import load_data, preprocess_data


def test_load_data():
    df = load_data('tests/utils/fakes/fake_data.csv')
    assert isinstance(df, pd.DataFrame)


def test_preprocess_data():
    df = pd.DataFrame({
        'satisfaction': ['satisfied', 'dissatisfied'],
        'Customer Type': ['Loyal Customer', 'Disloyal Customer'],
        'Age': [30, 40],
        'Type of Travel': ['Business travel', 'Personal travel'],
        'Class': ['Business', 'Eco'],
        'Flight Distance': [400, 500],
        'Ease of Online booking': [3, 4],
        'Online boarding': [4, 5]
    })

    X, y = preprocess_data(df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'satisfaction' not in X.columns
    assert y.equals(pd.Series([1, 0]))
