from backend.utils.model_evaluator import evaluate_model, format_metrics
import pandas as pd


def test_evaluate_model():
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    })
    y = pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    params = {'objective': 'binary:logistic'}
    metrics_storage, formatted_metrics = evaluate_model(X, y, params)

    for metrics in [metrics_storage, formatted_metrics]:
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics

        for metric in ['auc', 'precision', 'recall', 'f1']:
            for value in metrics[metric]:
                assert isinstance(value, (float, str)), (
                    f"Expected float or str, got {type(value)}"
                )


def test_evaluate_model_invalid_types():
    X = [1, 2, 3]
    y = [0, 1, 0]
    params = {'objective': 'binary:logistic'}
    try:
        evaluate_model(X, y, params)
    except ValueError:
        assert True, "Should raise a ValueError for invalid types."
    else:
        assert False, "Should have raised a ValueError."


def test_evaluate_model_mismatched_sizes():
    X = pd.DataFrame({'feature1': [1, 2, 3]})
    y = pd.Series([0, 1])
    params = {'objective': 'binary:logistic'}
    try:
        evaluate_model(X, y, params)
    except ValueError:
        assert True, "Should raise a ValueError for mismatched sizes."
    else:
        assert False, "Should have raised a ValueError."


def test_evaluate_model_invalid_params():
    X = pd.DataFrame({'feature1': [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    params = {'invalid_param': 'value'}
    try:
        evaluate_model(X, y, params)
    except ValueError:
        assert True, "Should raise a ValueError for invalid params."
    else:
        assert False, "Should have raised a ValueError."


def test_evaluate_model_empty_data():
    X = pd.DataFrame()
    y = pd.Series()
    params = {'objective': 'binary:logistic'}
    try:
        evaluate_model(X, y, params)
    except ValueError:
        assert True, "Should raise a ValueError for empty data."
    else:
        assert False, "Should have raised a ValueError."


def test_format_metrics():
    metrics = {
        'auc': [0.9, 0.91, 0.92],
        'precision': [0.87, 0.88, 0.89],
        'recall': [0.79, 0.8, 0.81],
        'f1': [0.83, 0.84, 0.85]
    }

    expected_result = {
        'auc': '0.910 ± 0.008',
        'precision': '0.880 ± 0.008',
        'recall': '0.800 ± 0.008',
        'f1': '0.840 ± 0.008'
    }

    formatted = format_metrics(metrics)
    assert formatted == expected_result, (
        f"Expected {expected_result}, got {formatted}"
    )
