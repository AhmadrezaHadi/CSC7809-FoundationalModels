from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true, y_pred):
        error = (y_true - y_pred) ** 2
        return np.mean(error)

    def derivative(self, y_true, y_pred):
        n = y_true.shape[0]
        return (-2 * (y_true - y_pred)) / n



class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        pass

