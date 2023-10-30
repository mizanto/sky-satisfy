import os
import logging

from flask import Flask, request, jsonify, abort, render_template_string
from src.utils.model_serializer import load_model
from src.utils.predictor import make_prediction
from src.utils.model_evaluator import format_metrics
from src.utils.metrics_storage import load_metrics, get_metrics_creation_date
from src.api_docs_generator import generate_api_endpoints_info
from src.train_and_save_model import train_and_save_model
from src.schemas.prediction_schema import PredictionSchema
from src.config import MODEL_FILE_PATH, METRICS_FILE_PATH


from marshmallow import ValidationError


try:
    if not os.path.exists(MODEL_FILE_PATH) or \
       not os.path.exists(METRICS_FILE_PATH):
        logging.error("Model or metrics not found. Starting training...")
        train_and_save_model('data/data.csv', 'models/')

    model = load_model()
    metrics = load_metrics()
except Exception as e:
    logging.error(f"An error occurred while loading the model or metrics: {e}")
    abort(500, description="Internal Server Error")


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests, using model to predict.
    """
    try:
        schema = PredictionSchema()
        data = schema.load(request.json)
        prediction = make_prediction(model, data)
        return jsonify({'prediction': prediction})
    except ValidationError as e:
        return jsonify({'error': str(e.normalized_messages())}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Provide information about the model, including its type and metrics.
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
    Health check route. Returns 'OK' if the API is up and running.
    """
    return jsonify({'status': 'OK'}), 200


@app.route('/', methods=['GET'])
def api_info():
    """
    Provide a list of available endpoints and their documentation.
    """
    docs = generate_api_endpoints_info()
    return render_template_string(docs)


if __name__ == '__main__':
    app.run(debug=True)
