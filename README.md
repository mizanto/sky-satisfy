# Sky Satisfy: Airline Customer Satisfaction Prediction

## Description

This project is a part of the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course. It utilizes machine learning to predict customer satisfaction for airlines based on various factors such as class, flight distance, and in-flight entertainment. The model is trained on a [Kaggle dataset](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline/data) that includes 129,880 customer records.

## Technologies Used

- Python: 3.11
- NumPy: 1.26.1
- scikit-learn: 1.3.2
- Flask: 3.0.0
- pandas: 2.1.2
- XGBoost: 2.0.1
- pytest: 7.4.3
- marshmallow: 3.20.1
- gunicorn: 21.2.0
- flasgger: 0.9.7.1

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/mizanto/sky-satisfy.git
    ```

2. Navigate to the project directory:

    ```bash
    cd sky-satisfy
    ```

3. Install dependencies from `Pipfile`:

    ```bash
    pipenv install
    ```

## Usage

### Model Training

You can train the model by running the following command:

```bash
python src/train_model.py
```

### Running the Service

Run the service, and if the model is missing in the `models` folder, the service will automatically train it:

```bash
gunicorn --bind 0.0.0.0:8000 src.prediction_service:app
```

### Running Tests

To run the tests, use the following command:

```bash
pytest
```

## Docker

To run the application in Docker, execute the following commands:

1. Build the image:

    ```bash
    docker build -t sky-satisfy .
    ```

2. Run the container:

    ```bash
    docker run -p 8000:8000 sky-satisfy
    ```

## Project Structure

```
.
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── README.md
├── data/
│   └── data.csv
├── models/
|   ├── metrics.json
│   └── model.pkl
├── notebooks/
|   └── data_exploration.ipynb
├── src/
|   ├── config.py
|   ├── prediction_service.py
|   ├── train_and_save_model.py
│   └── utils/
|       ├── data_loader.py
|       ├── metrics_storage.py
|       ├── model_evaluator.py
|       ├── model_serializer.py
|       ├── model_trainer.py
|       └── predictor.py
└── tests/
    ├── test_prediction_service.py
    └── utils/
        ├── test_data_loader.py
        ├── test_metrics_storage.py
        ├── test_model_evaluator.py
        ├── test_model_serializer.py
        ├── test_model_trainer.py
        ├── test_predictor.py
        └── fakes/
```

## Author

[Sergei Bendak](https://github.com/mizanto)
