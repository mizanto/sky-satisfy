"""
Utility functions for loading and preprocessing data for the
SkySatisfy project.
"""

import pandas as pd


COLUMNS_TO_KEEP = [
    'satisfaction',
    'Customer Type',
    'Age',
    'Type of Travel',
    'Class',
    'Flight Distance',
    'Ease of Online booking',
    'Online boarding'
]

TARGET_COLUMN = 'satisfaction'


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """Preprocess the data and return features and target variable."""
    df = df.copy()
    df = df[COLUMNS_TO_KEEP]
    features = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[features]
    y = df[TARGET_COLUMN]
    return X, y
