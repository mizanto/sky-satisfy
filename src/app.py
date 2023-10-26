from flask import Flask, request, jsonify, abort, render_template_string
from src.utils.model_serializer import load_model
from src.utils.predictor import make_prediction
from src.utils.model_evaluator import format_metrics
from src.utils.metrics_storage import load_metrics, get_file_creation_date
from src.api_docs_generator import generate_api_endpoints_info
from marshmallow import Schema, fields, ValidationError, validate

PATH_TO_MODEL = 'models/model.pkl'
PATH_TO_METRICS = 'models/metrics.json'

try:
    model = load_model(PATH_TO_MODEL)
    metrics = load_metrics(PATH_TO_METRICS)
except Exception as e:
    print(f"An error occurred while loading the model or metrics: {e}")
    abort(500, description="Internal Server Error")


app = Flask(__name__)


class PredictionSchema(Schema):
    customer_type = fields.Str(validate=validate.OneOf(['loyal_customer',
                                                        'disloyal_customer']))
    age = fields.Int(validate=validate.Range(min=0, max=120))
    type_of_travel = fields.Str(validate=validate.OneOf(['business_travel',
                                                         'personal_travel']))
    flight_distance = fields.Int(required=True)
    ease_of_online_booking = fields.Int(validate=validate.Range(min=0, max=5))
    online_boarding = fields.Int(validate=validate.Range(min=0, max=5))
    class_ = fields.Str(
        data_key='class', validate=validate.OneOf(['business',
                                                   'eco',
                                                   'eco_plus']))


@app.route('/predict', methods=['POST'])
def predict():
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
    formatted_metrics = format_metrics(metrics)
    try:
        info = {
            'model_type': 'XGBoost',
            'training_date': get_file_creation_date(PATH_TO_MODEL),
            'metrics': formatted_metrics
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'OK'}), 200


@app.route('/', methods=['GET'])
def api_info():
    docs = generate_api_endpoints_info()
    return render_template_string(docs)


if __name__ == '__main__':
    app.run(debug=True)
