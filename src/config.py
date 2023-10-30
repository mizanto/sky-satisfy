import os


# path to the dataset file
DATASET_FILE_PATH = 'data/data.csv'

# path to the model folder
MODEL_FOLDER_PATH = 'models/'

# path to the model file
MODEL_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, 'model.pkl')

# path to the metrics file
METRICS_FILE_PATH = os.path.join(MODEL_FOLDER_PATH, 'metrics.json')

# model parameters
MODEL_PARAMS = {
    'XGB_PARAMS': {
        'eta': 0.3,
        'max_depth': 6,
        'min_child_weight': 10,
        'objective': 'binary:logistic',
        'nthread': 8,
        'seed': 42,
    },
    'XGB_NUM_BOOST_ROUND': 25,
}
