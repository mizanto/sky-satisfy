from src.utils.model_evaluator import evaluate_model
import pandas as pd


def test_evaluate_model():
    X = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'feature2': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]})
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    params = {'objective': 'binary:logistic'}
    metrics = evaluate_model(X, y, params)
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 'auc' in metrics

    for metric in ['auc', 'precision', 'recall', 'f1']:
        for value in metrics[metric]:
            assert isinstance(value, float)
