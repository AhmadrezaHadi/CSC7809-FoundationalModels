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
        pass

    def derivative(self, y_true, y_pred):
        pass


class CrossEntropy(LossFunction):
    def loss(self, y_true, y_pred):
        pass

    def derivative(self, y_true, y_pred):
        pass

