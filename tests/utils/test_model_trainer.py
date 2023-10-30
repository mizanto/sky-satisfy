import pandas as pd
import xgboost as xgb
from src.utils.model_trainer import train_model


def test_train_model():
    X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [3, 4, 5]})
    y = pd.Series([1, 0, 1])
    params = {'objective': 'binary:logistic'}

    model = train_model(X, y, params)
    assert isinstance(model, xgb.Booster), \
        "Trained model should be an instance of xgb.Booster"
