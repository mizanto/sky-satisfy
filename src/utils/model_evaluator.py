"""
Utility functions for evaluating machine learning models for
the SkySatisfy project.
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import (roc_auc_score,
                             precision_score,
                             recall_score,
                             f1_score)
import xgboost as xgb


def evaluate_model(X: pd.DataFrame, y: pd.Series, xgb_params: dict) -> dict:
    """Evaluate the model and return metrics."""
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

    return metrics_storage
