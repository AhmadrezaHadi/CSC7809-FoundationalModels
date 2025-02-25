from abc import ABC, abstractmethod
import numpy as np



class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x):
        result = 1 / (1 + np.exp(-x))
        return result
    
    def derivative(self, x):
        sig_x = self.forward(x)
        result = sig_x * (1 - sig_x)
        return result


class Tanh(ActivationFunction):
    def forward(self, x):
        return  np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU(ActivationFunction):
    def forward(self, x):
        result = np.maximum(0, x)
        return result
    
    def derivative(self, x):
        result = np.where(x > 0, 1, 0)
        return result


class Linear(ActivationFunction):
    def forward(self, x):
        return x
    
    def derivative(self, x):
        return np.ones_like(x)


class Softplus(ActivationFunction):
    def forward(self, x):
        return np.log(1 + np.exp(x))

    def derivative(self, x):
        return 1 / (1 + np.exp(-x)) 


class Mish(ActivationFunction):
    def forward(self, x):
        softplus_val = np.log(1 + np.exp(x))
        return x * np.tanh(softplus_val)

    def derivative(self, x):
        softplus_val = np.log(1 + np.exp(x))
        tanh_softplus = np.tanh(softplus_val)
        return tanh_softplus + x * (1 - tanh_softplus ** 2) * (1 / (1 + np.exp(-x)))


class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)  
