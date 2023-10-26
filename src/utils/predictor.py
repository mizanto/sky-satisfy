import pandas as pd
import xgboost as xgb


def make_prediction(model: xgb.Booster, data: dict) -> list:
    """Make a prediction using the model."""
    try:
        df = pd.DataFrame([data])
        dmatrix = xgb.DMatrix(df)
        prediction = model.predict(dmatrix)
        return prediction.tolist()
    except Exception as e:
        return {'error': str(e)}
