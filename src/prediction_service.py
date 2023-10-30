import os
import logging

from flask import Flask, request, jsonify, abort
from src.utils.model_serializer import load_model
from src.utils.predictor import make_prediction
from src.utils.model_evaluator import format_metrics
from src.utils.metrics_storage import load_metrics, get_metrics_creation_date
from src.train_and_save_model import train_and_save_model
from src.schemas.prediction_schema import PredictionSchema
from src.config import MODEL_FILE_PATH, METRICS_FILE_PATH

from marshmallow import ValidationError
from flasgger import Swagger


logging.basicConfig(level=logging.INFO)


try:
    if not os.path.exists(MODEL_FILE_PATH) or \
       not os.path.exists(METRICS_FILE_PATH):
        logging.error("Model or metrics not found. Starting training...")
        train_and_save_model()

    model = load_model()
    metrics = load_metrics()
except Exception as e:
    logging.error(f"An error occurred while loading the model or metrics: {e}")
    abort(500, description="Internal Server Error")


app = Flask(__name__)
swagger = Swagger(app)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction based on input data
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          id: PredictionInput
          required:
            - customer_type
            - age
            - type_of_travel
            - flight_distance
            - ease_of_online_booking
            - online_boarding
            - class
          properties:
            customer_type:
              type: string
              enum: ["loyal_customer", "disloyal_customer"]
              description: The type of the customer
              required: true
            age:
              type: integer
              minimum: 0
              maximum: 120
              description: The age of the customer
              required: true
            type_of_travel:
              type: string
              enum: ["business_travel", "personal_travel"]
              description: The type of travel
              required: true
            flight_distance:
              type: integer
              description: The distance of the flight in miles
              required: true
            ease_of_online_booking:
              type: integer
              minimum: 0
              maximum: 5
              description: Ease of online booking, rated between 0 and 5
              required: true
            online_boarding:
              type: integer
              minimum: 0
              maximum: 5
              description: Online boarding experience, rated between 0 and 5
              required: true
            class:
              type: string
              enum: ["business", "eco", "eco_plus"]
              description: Class of the flight
              required: true
    responses:
      200:
        description: Prediction result
        schema:
          id: PredictionOutput
          properties:
            prediction:
              type: number
              format: float
              description: The prediction score
            verdict:
              type: string
              description: The verdict based on the prediction
        examples:
          {
            "prediction": 0.991,
            "verdict": "satisfied"
          }
      400:
        description: Invalid input
        examples:
          {
            "error": "Invalid input data"
          }
    """
    try:
        schema = PredictionSchema()
        data = schema.load(request.json)
        prediction = make_prediction(model, data)

        verdict = "satisfied" if prediction > 0.5 else "Not satisfied"

        return jsonify({
            'prediction': round(prediction, 3),
            'verdict': verdict
          })
    except ValidationError as e:
        logging.error(f"Validation error: {e.normalized_messages()}")
        return jsonify({'error': str(e.normalized_messages())}), 400
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return jsonify({'error': str(e)}), 400


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Retrieve model information and metrics
    ---
    responses:
      200:
        description: Information about the model
        schema:
          id: ModelInfoOutput
          properties:
            metrics:
              type: object
              properties:
                auc:
                  type: string
                f1:
                  type: string
                precision:
                  type: string
                recall:
                  type: string
            model_type:
              type: string
            training_date:
              type: string
    """
    formatted_metrics = format_metrics(metrics)
    try:
        info = {
            'model_type': 'XGBoost',
            'training_date': get_metrics_creation_date(MODEL_FILE_PATH),
            'metrics': formatted_metrics
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    """
    Check the health status of the API
    ---
    responses:
      200:
        description: API is healthy
        schema:
          id: HealthCheckOutput
          properties:
            status:
              type: string
              description: The health status of the API
    """
    return jsonify({'status': 'OK'}), 200


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
