import pandas as pd
import pytest

from backend.utils.data_loader import load_data, preprocess_data
from backend.config import DATASET_FILE_PATH


def test_load_data():
    df = load_data(DATASET_FILE_PATH)
    assert isinstance(df, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(df)}"
    )


def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data('nonexistent_file.csv')


def test_load_data_empty_file():
    with pytest.raises(pd.errors.EmptyDataError):
        load_data('tests/utils/fakes/empty_file.csv')


def test_preprocess_data_missing_columns():
    df = pd.DataFrame({'some_column': [1, 2, 3]})
    with pytest.raises(KeyError):
        preprocess_data(df)


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
    assert isinstance(X, pd.DataFrame), (
        f"Expected pd.DataFrame, got {type(X)}"
    )
    assert isinstance(y, pd.Series), (
        f"Expected pd.Series, got {type(y)}"
    )
    assert 'satisfaction' not in X.columns, (
        "'satisfaction' should not be in X.columns"
    )
    assert y.equals(pd.Series([1, 0])), (
        "y should be equal to pd.Series([1, 0])"
    )
