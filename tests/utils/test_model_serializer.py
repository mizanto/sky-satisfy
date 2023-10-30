import pandas as pd
import pytest
import xgboost as xgb
from src.utils.model_serializer import save_model, load_model
from src.utils.model_trainer import train_model


def test_model_serializer():
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [3, 4, 5]})
    y = pd.Series([1, 0, 1])
    params = {'objective': 'binary:logistic'}
    model = train_model(X, y, params)
    save_model(model, path='./tests/utils/fakes/test_model.pkl')

    loaded_model = load_model()
    assert isinstance(loaded_model, xgb.Booster), (
        f"Expected xgb.Booster, got {type(loaded_model)}"
    )


def test_save_model_invalid_type():
    model = "not_a_model"
    try:
        save_model(model, path='./tests/utils/fakes/test_invalid_model.pkl')
    except TypeError:
        assert True, "Should raise a TypeError for invalid model type."
    else:
        assert False, "Should have raised a TypeError."


def test_load_model_invalid_path():
    try:
        load_model(path='./tests/utils/fakes/non_existent_model.pkl')
    except FileNotFoundError:
        assert True, "Should raise a FileNotFoundError for invalid file path."
    else:
        assert False, "Should have raised a FileNotFoundError."


def test_load_model_empty_file():
    try:
        load_model(path='./tests/utils/fakes/empty_model.pkl')
    except EOFError:
        assert True, "Should raise an EOFError for empty file."
    else:
        assert False, "Should have raised an EOFError."


def test_model_functionality_after_loading():
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [3, 4, 5]})
    y = pd.Series([1, 0, 1])
    params = {'objective': 'binary:logistic'}
    original_model = train_model(X, y, params)
    save_model(original_model, path='./tests/utils/fakes/test_model.pkl')

    loaded_model = load_model(path='./tests/utils/fakes/test_model.pkl')
    original_pred = original_model.predict(xgb.DMatrix(X))
    loaded_pred = loaded_model.predict(xgb.DMatrix(X))

    assert all(original_pred == loaded_pred), (
        "Model should work the same after loading.")


def test_save_model_io_error():
    model = xgb.Booster()
    invalid_path = '/invalid_path/saved_model.pkl'

    with pytest.raises(IOError):
        save_model(model, path=invalid_path)
