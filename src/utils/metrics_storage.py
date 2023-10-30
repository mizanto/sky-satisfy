import json
import logging
import logging.config
import os
import time

from src.config import METRICS_FILE_PATH, LOGGING_CONFIG


logging.config.dictConfig(LOGGING_CONFIG)


def get_metrics_creation_date(file_path=METRICS_FILE_PATH) -> str:
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


def save_metrics(metrics: dict, path=METRICS_FILE_PATH):
    """
    Save metrics to a JSON file.

    Parameters:
    - metrics (dict): Metrics to save.
    - path (str): File path to save the metrics.

    Example:
    >>> save_metrics({'accuracy': 0.9}, 'metrics.json')
    """
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f)
        logging.info(f"Successfully saved metrics to {path}")
    except Exception as e:
        logging.error(f"Failed to save metrics: {e}")
        raise


def load_metrics(path=METRICS_FILE_PATH) -> dict:
    """
    Load metrics from a JSON file.

    Parameters:
    - path (str): File path to load the metrics from.

    Returns:
    - dict: Loaded metrics.

    Example:
    >>> metrics = load_metrics('metrics.json')
    """
    if not os.path.exists(path):
        logging.error(f"Metrics file {path} does not exist.")
        raise FileNotFoundError(f"{path} does not exist.")
    try:
        with open(path, 'r') as f:
            metrics = json.load(f)
        logging.info(f"Successfully loaded metrics from {path}")
        return metrics
    except Exception as e:
        logging.error(f"Failed to load metrics: {e}")
        raise
