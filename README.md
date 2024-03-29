# Sky Satisfy: Airline Customer Satisfaction Prediction

## Description

This [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) project focuses on preemptively predicting airline customer satisfaction using pre-flight data from [Kaggle](https://www.kaggle.com/datasets/yakhyojon/customer-satisfaction-in-airline/data). The model is designed to analyze factors available before the flight takes off, such as ease of online booking, online boarding, seating class, customer type, age, travel type, and flight distance. By leveraging these inputs, the model aims to help airlines identify potential dissatisfaction risks and improve customer experience proactively, ensuring that interventions can be made before the customer even steps onto the airplane.

## API Endpoints

#### /apidocs

`GET` request to view the detailed API documentation.

#### /predict

`POST` request to make a prediction based on input data.

#### /model/info

`GET` request to retrieve model information and metrics.

#### /health

`GET` request to check the health status of the API.

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

### Local Setup

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

### Docker Setup (Local Build)

Alternatively, you can build the Docker image locally:

Build the image:

```bash
docker build -t sky-satisfy .
```

### Docker Setup from Docker Hub

You can also pull the image from Docker Hub:

Pull the image:

```bash
docker pull sergben/sky-satisfy:v1.0.1
```

## Usage

### Model Training

You can train the model by running the following command:

```bash
python backend/train_model.py
```

## Running the Service

### Local Run

Run the service locally, and if the model is missing in the `models` folder, the service will automatically train it:

```bash
gunicorn --bind 0.0.0.0:8000 backend.prediction_service:app
```

### Docker Run

#### Local Build

If you have built the Docker image locally, you can run the service as follows:

```bash
docker run -d -p 8000:8000 sky-satisfy
```

#### Docker Hub

If you have pulled the image from Docker Hub, you can run the service as follows:

```bash
docker run -d -p 80:80 sergben/sky-satisfy:v1.0.1
```

### Running Tests

To run the tests, use the following command:

```bash
pytest
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
├── backend/
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
├── frontend/
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css
│   │   ├── js/
│   │   │   └── main.js
│   │
│   └── templates/
│       └── index.html
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

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
