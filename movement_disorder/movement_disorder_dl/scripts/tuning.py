from ..tuner.hyperparameter_tuner import tune_cnn_1d

def tune_model():
    result_grid = tune_cnn_1d()
    return result_grid