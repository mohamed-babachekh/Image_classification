import numpy as np


class EstimatorInterface:
    def fit(self, x_train: np.array, y_train:  np.array) -> object:
        pass

    def predict(self, x_test:  np.array) ->  np.array:
        pass

    @staticmethod
    def save(model: object, path: str = 'model.joblib'):
        pass

    @staticmethod
    def load(model_path: str):
        pass
