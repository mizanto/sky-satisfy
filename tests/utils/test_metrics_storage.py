import json
import os
import re
from backend.utils.metrics_storage import (save_metrics, load_metrics,
                                       get_metrics_creation_date)


def test_save_metrics():
    metrics = {'accuracy': 0.9}
    path = 'test_metrics.json'

    if os.path.exists(path):
        os.remove(path)

    save_metrics(metrics, path)

    assert os.path.exists(path), "Metrics file should exist after saving."

    with open(path, 'r') as f:
        saved_metrics = json.load(f)

    assert saved_metrics == metrics, "Saved metrics should match the original."

    os.remove(path)


def test_load_metrics():
    metrics = {'accuracy': 0.9}
    path = 'test_metrics.json'

    with open(path, 'w') as f:
        json.dump(metrics, f)

    loaded_metrics = load_metrics(path)

    assert loaded_metrics == metrics, (
        "Loaded metrics should match the saved metrics.")

    os.remove(path)


def test_get_metrics_creation_date():
    path = 'test_metrics.json'
    metrics = {'accuracy': 0.9}

    with open(path, 'w') as f:
        json.dump(metrics, f)

    creation_date = get_metrics_creation_date(path)

    assert re.match(r'\d{4}-\d{2}-\d{2}', creation_date), (
        "Creation date should be in 'YYYY-MM-DD' format.")

    os.remove(path)


def test_error_handling():
    non_existent_path = 'non_existent_metrics.json'

    try:
        load_metrics(non_existent_path)
    except FileNotFoundError:
        assert True, "Should raise a FileNotFoundError for non-existent files."
    else:
        assert False, "Should have raised a FileNotFoundError."
