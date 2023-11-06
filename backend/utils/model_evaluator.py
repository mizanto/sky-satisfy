"""
Utility functions for evaluating machine learning models for
the SkySatisfy project.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score)
import xgboost as xgb
from backend.config import MODEL_PARAMS


def evaluate_model(X: pd.DataFrame, y: pd.Series,
                   xgb_params=MODEL_PARAMS['XGB_PARAMS']) -> (dict, dict):
    """
    Evaluate an XGBoost model using 5-fold cross-validation.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target vector.
    - xgb_params (dict): Parameters for the XGBoost model.

    Returns:
    - tuple: Raw metrics and formatted metrics.

    Example:
    >>> X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    >>> y = pd.Series([0, 1, 0])
    >>> params = {'eta': 0.3, 'max_depth': 6}
    >>> raw_metrics, formatted_metrics = evaluate_model(X, y, params)
    """
    metrics_storage = {
        'auc': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    dtrain = xgb.DMatrix(X, label=y)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_index, val_index in kf.split(y):
        dtrain_fold = dtrain.slice(train_index)
        dval_fold = dtrain.slice(val_index)

        model = xgb.train(xgb_params, dtrain_fold, num_boost_round=25)

        y_pred_proba = model.predict(dval_fold)
        y_true = dtrain.get_label()[val_index]

        metrics_storage['auc'].append(roc_auc_score(y_true, y_pred_proba))
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics_storage['precision'].append(
            precision_score(y_true, y_pred, zero_division=1))
        metrics_storage['recall'].append(recall_score(y_true, y_pred))
        metrics_storage['f1'].append(f1_score(y_true, y_pred))

    formatted_metrics = format_metrics(metrics_storage)

    return metrics_storage, formatted_metrics


def format_metrics(metrics: dict) -> str:
    formatted_metrics = {}
    for key, values in metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        formatted_metrics[key] = f"{mean_value:.3f} \u00B1 {std_value:.3f}"
    return formatted_metrics
