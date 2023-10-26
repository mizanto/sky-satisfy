"""
Utility functions for training machine learning models for
the SkySatisfy project.
"""

import pandas as pd
import xgboost as xgb


def train_model(X: pd.DataFrame, y: pd.Series, params: dict) -> xgb.Booster:
    """Train an XGBoost model."""
    dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
    model = xgb.train(params, dtrain, num_boost_round=25)
    return model
