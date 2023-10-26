"""
Script for training, evaluating, and saving the flight satisfaction model for
the SkySatisfy project.
"""

from utils.data_loader import load_data, preprocess_data
from utils.model_trainer import train_model
from utils.model_evaluator import evaluate_model
from utils.model_serializer import save_model

XGB_PARAMS = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'nthread': 8,
    'seed': 42,
}


def main(data_path: str, model_save_path: str):
    df = load_data(data_path)
    X, y = preprocess_data(df)
    model = train_model(X, y, XGB_PARAMS)
    metrics = evaluate_model(model, X, y)
    print(metrics)
    save_model(model, model_save_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(
            "Usage: python train_and_save_model.py path_to_csv path_to_model")
    else:
        main(sys.argv[1], sys.argv[2])
