import pandas as pd
import xgboost as xgb
import logging
import logging.config

from backend.config import LOGGING_CONFIG


logging.config.dictConfig(LOGGING_CONFIG)


def make_prediction(model: xgb.Booster, data: dict) -> float:
    """
    Make a prediction using an XGBoost model.

    Parameters:
    - model (xgb.Booster): Trained XGBoost model.
    - data (dict): Data for prediction.

    Returns:
    - float: Prediction result.
    """
    try:
        # Convert the input data to a DataFrame
        df = pd.DataFrame([data])

        # Prepare the data for prediction
        df = _prepare_data(df)

        # Convert the DataFrame to DMatrix, which is required by XGBoost
        dmatrix = xgb.DMatrix(df)

        # Make the prediction
        prediction = model.predict(dmatrix)[0]
        logging.info(f"Prediction made: {prediction}")

        # Return the prediction as a float
        return float(prediction)
    except Exception as e:
        logging.error(f"Error in make_prediction: {e}")
        raise e  # Re-raise the exception to handle it


def _prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for prediction by encoding categorical variables and ensuring
    all expected columns are present.

    Parameters:
    - df (pd.DataFrame): The input data frame with raw data.

    Returns:
    - pd.DataFrame: The processed data frame ready for prediction.
    """

    try:
        # If the data has the column named 'class_', rename it to 'class'
        if 'class_' in df.columns:
            df.rename(columns={'class_': 'class'}, inplace=True)

        # Encode 'customer_type' and 'type_of_travel' as integers
        df['customer_type'] = df['customer_type'].map({
            'loyal_customer': 1,
            'disloyal_customer': 0
        })
        df['type_of_travel'] = df['type_of_travel'].map({
            'business_travel': 1,
            'personal_travel': 0
        })

        # Create dummy variables for the 'class' column
        class_dummies = pd.get_dummies(df['class'], prefix='class')

        # Drop the original 'class' column
        df = df.drop('class', axis=1)

        # Concatenate the dummy columns
        df = pd.concat([df, class_dummies], axis=1)

        # Ensure the DF has all the expected columns, in the correct order
        expected_columns = ['customer_type', 'age', 'type_of_travel',
                            'flight_distance', 'ease_of_online_booking',
                            'online_boarding', 'class_business', 'class_eco',
                            'class_eco_plus']
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Convert all columns to integer type
        df = df.astype(int)

        return df
    except Exception as e:
        logging.error(f"Error in _prepare_data: {e}")
        raise e  # Re-raise the exception to handle it further
