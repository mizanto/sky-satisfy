"""
Utility functions for loading and preprocessing data for the
SkySatisfy project.
"""

import pandas as pd


COLUMNS_TO_KEEP = [
    'satisfaction',
    'customer_type',
    'age',
    'type_of_travel',
    'class',
    'flight_distance',
    'ease_of_online_booking',
    'online_boarding'
]

TARGET_COLUMN = 'satisfaction'
FEATURES = [c for c in COLUMNS_TO_KEEP if c != TARGET_COLUMN]


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Parameters:
    - data_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded data.

    Example:
    >>> df = load_data('data.csv')
    """
    return pd.read_csv(data_path)


def preprocess_data(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    """
    Preprocess the data and return features and target variable.

    Parameters:
    - df (pd.DataFrame): Dataframe to preprocess.

    Returns:
    - tuple: Feature matrix (pd.DataFrame) and target vector (pd.Series).

    Example:
    >>> X, y = preprocess_data(df)
    """
    df = df.copy()
    df = _preprocess_column_names(df)

    df = df[COLUMNS_TO_KEEP]
    df = _encode_categorical_features(df)

    X = df[[c for c in df.columns if c != TARGET_COLUMN]]
    y = df[TARGET_COLUMN]

    return X, y


def _preprocess_column_names(df):
    ''' Convert column names to lowercase & replace spaces with underscores'''
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    string_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    return df


def _encode_categorical_features(df):
    """Replace string labels with numerical labels."""
    df = df.copy()
    df['satisfaction'] = df['satisfaction'].replace({'satisfied': 1,
                                                     'dissatisfied': 0})
    df['customer_type'] = df['customer_type'].replace({'loyal_customer': 1,
                                                       'disloyal_customer': 0})
    df['type_of_travel'] = df['type_of_travel'].replace({'business_travel': 1,
                                                         'personal_travel': 0})
    df = pd.get_dummies(df, columns=['class'], prefix='class')
    return df
