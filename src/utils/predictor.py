import pandas as pd
import xgboost as xgb
import logging
import logging.config

from src.config import LOGGING_CONFIG


logging.config.dictConfig(LOGGING_CONFIG)


def make_prediction(model: xgb.Booster, data: dict) -> list:
    """
    Make a prediction using an XGBoost model.

    Parameters:
    - model (xgb.Booster): Trained XGBoost model.
    - data (dict): Data for prediction.

    Returns:
    - list: Prediction result.

    Example:
    >>> model = xgb.Booster(model_file='model.json')
    >>> data = {'feature1': 1, 'feature2': 4}
    >>> prediction = make_prediction(model, data)
    """
    try:
        df = pd.DataFrame([data])
        df = _prepare_data(df)
        dmatrix = xgb.DMatrix(df)
        prediction = model.predict(dmatrix)[0]
        logging.info(f"Prediction made: {prediction}")
        return float(prediction)
    except Exception as e:
        logging.error(f"Error in make_prediction: {e}")
        return {'error': str(e)}


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for prediction.
    """
    # Rename & encode columns
    df.rename(columns={'class_': 'class'}, inplace=True)
    df['customer_type'] = df['customer_type'].replace({
        'loyal_customer': 1,
        'disloyal_customer': 0
    })
    df['type_of_travel'] = df['type_of_travel'].replace({
        'business_travel': 1,
        'personal_travel': 0
    })
    df = pd.get_dummies(df, columns=['class'], prefix='class')
    df = df.astype(int)

    # Add missing columns
    expected_columns = ['customer_type', 'age', 'type_of_travel',
                        'flight_distance', 'ease_of_online_booking',
                        'online_boarding', 'class_business', 'class_eco',
                        'class_eco_plus']

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df
