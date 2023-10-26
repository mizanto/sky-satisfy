import json
import os
import time


def get_file_creation_date(file_path: str) -> str:
    """
    Return the creation date of a file.

    Parameters:
    - file_path (str): Path to the file.

    Returns:
    - str: Creation date in 'YYYY-MM-DD' format.

    Example:
    >>> date = get_file_creation_date('metrics.json')
    """
    file_stat = os.stat(file_path)
    return time.strftime('%Y-%m-%d', time.localtime(file_stat.st_ctime))


def save_metrics(metrics: dict, path: str):
    """
    Save metrics to a JSON file.

    Parameters:
    - metrics (dict): Metrics to save.
    - path (str): File path to save the metrics.

    Example:
    >>> save_metrics({'accuracy': 0.9}, 'metrics.json')
    """
    with open(path, 'w') as f:
        json.dump(metrics, f)


def load_metrics(path: str) -> dict:
    """
    Load metrics from a JSON file.

    Parameters:
    - path (str): File path to load the metrics from.

    Returns:
    - dict: Loaded metrics.

    Example:
    >>> metrics = load_metrics('metrics.json')
    """
    with open(path, 'r') as f:
        return json.load(f)
