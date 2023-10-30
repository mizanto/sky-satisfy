"""
Utility functions for training machine learning models for
the SkySatisfy project.
"""

import pandas as pd
import xgboost as xgb
from src.config import MODEL_PARAMS


def train_model(X: pd.DataFrame, y: pd.Series,
                params=MODEL_PARAMS['XGB_PARAMS']) -> xgb.Booster:
    """
    Train an XGBoost model.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - params (dict): Parameters for the XGBoost model.

    Returns:
    - xgb.Booster: Trained XGBoost model.

    Example:
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> y = pd.Series([0, 1, 0])
    >>> params = {'eta': 0.3, 'max_depth': 6}
    >>> model = train_model(X, y, params)
    """
    dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
    model = xgb.train(params, dtrain,
                      num_boost_round=MODEL_PARAMS['XGB_NUM_BOOST_ROUND'])
    return model
