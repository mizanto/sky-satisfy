"""
Utility functions for saving and loading machine learning models for
the SkySatisfy project.
"""

import pickle
import xgboost as xgb


def save_model(model: xgb.Booster, path: str):
    """Save the model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path: str) -> xgb.Booster:
    """Load the model from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)
