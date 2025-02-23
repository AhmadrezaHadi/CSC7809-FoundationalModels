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
        return 0.5 * np.mean((y_true - y_pred) ** 2)

    def derivative(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.shape[0]



class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        # epsilon = 1e-12
        # y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_true, y_pred):
        # epsilon = 1e-12
        # y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        batch_size = y_true.shape[0]
        return -(y_true / y_pred) / batch_size
