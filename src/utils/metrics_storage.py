import json
import os
import time


def get_file_creation_date(file_path: str) -> str:
    """Return the creation date of a file."""
    file_stat = os.stat(file_path)
    return time.strftime('%Y-%m-%d', time.localtime(file_stat.st_ctime))


def save_metrics(metrics: dict, path: str):
    """Save metrics to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f)


def load_metrics(path: str) -> dict:
    """Load metrics from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)
