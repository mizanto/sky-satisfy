"""
Utility functions for saving and loading machine learning models for
the SkySatisfy project.
"""

import pickle
import xgboost as xgb


def save_model(model: xgb.Booster, path: str):
    """
    Save the XGBoost model to disk.

    Parameters:
    - model (xgb.Booster): Trained XGBoost model.
    - path (str): File path to save the model.

    Example:
    >>> model = xgb.Booster(model_file='model.json')
    >>> save_model(model, 'saved_model.pkl')
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> xgb.Booster:
    """
    Load the XGBoost model from disk.

    Parameters:
    - path (str): File path to load the model from.

    Returns:
    - xgb.Booster: Loaded XGBoost model.

    Example:
    >>> model = load_model('saved_model.pkl')
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
