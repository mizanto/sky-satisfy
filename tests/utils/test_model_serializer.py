import pandas as pd
import xgboost as xgb
from src.utils.model_serializer import save_model, load_model
from src.utils.model_trainer import train_model


def test_model_serializer():
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [3, 4, 5]})
    y = pd.Series([1, 0, 1])
    params = {'objective': 'binary:logistic'}
    model = train_model(X, y, params)
    save_model(model)

    loaded_model = load_model()
    assert isinstance(loaded_model, xgb.Booster), (
        f"Expected xgb.Booster, got {type(loaded_model)}"
    )
