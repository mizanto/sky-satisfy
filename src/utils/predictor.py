import pandas as pd
import xgboost as xgb


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
        dmatrix = xgb.DMatrix(df)
        prediction = model.predict(dmatrix)
        return prediction.tolist()
    except Exception as e:
        return {'error': str(e)}
