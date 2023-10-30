#!/usr/bin/env python3

"""
Script for training, evaluating, and saving the flight satisfaction model for
the SkySatisfy project.
"""
import logging

from src.utils.data_loader import load_data, preprocess_data
from src.utils.metrics_storage import save_metrics
from src.utils.model_trainer import train_model
from src.utils.model_evaluator import evaluate_model
from src.utils.model_serializer import save_model


def train_and_save_model():
    """
    Train a machine learning model and save it along with its metrics.

    This function takes in the path to the training data and a folder path
    where the trained model and metrics will be saved. It performs the
    following steps:
    1. Load the data from the given data path.
    2. Preprocess the data.
    3. Train the model using the preprocessed data.
    4. Evaluate the model and print the metrics.
    5. Save the trained model and metrics to the specified folder.

    Parameters:
    - data_path (str): The path to the training data.
    - model_folder_path (str): The folder path where the trained model and
                               metrics will be saved.

    Returns:
    None
    """
    # Load the data and preprocess it
    df = load_data()
    X, y = preprocess_data(df)
    model = train_model(X, y)
    raw_metrics, formatted_metrics = evaluate_model(X, y)
    logging.info(f"Model metrics: {raw_metrics}")

    # Save the model and metrics
    save_model(model)
    save_metrics(raw_metrics)


if __name__ == "__main__":
    train_and_save_model()
