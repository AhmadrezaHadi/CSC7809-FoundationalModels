import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple
from activation_functions import Sigmoid, Tanh, ReLU, Linear, Softmax, Mish, Linear, ActivationFunction
from utils import glorot_initialization, batch_generator
from loss_functions import LossFunction, SquaredError, CrossEntropy


class Layer:
    def __init__(
        self, 
        fan_in: int, 
        fan_out: int, 
        activation_function: ActivationFunction,
        dropout_rate: float = 0.0  
    ):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function
        self.dropout_rate = dropout_rate
        self.mask = None
        self.activations = None
        self.delta = None
        self.z = None

        # Initialize weights and biaes
        self.W = glorot_initialization(fan_in, fan_out)
        self.b = np.zeros((1, fan_out))

    def forward(self, h: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Computes the activations for this layer

        :param h: input to layer
        :param training: if True, apply dropout
        :return: layer activations
        """
        self.z = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(self.z)

        if training and self.dropout_rate > 0.0:
            keep_prob = 1.0 - self.dropout_rate
            self.mask = (np.random.rand(*self.activations.shape) < keep_prob)
            self.activations = self.activations * self.mask / keep_prob
        else:
            self.mask = None

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param h: input to this layer from forward pass
        :param delta: dL/da for this layer
        :return: (dW, dB)
        """
        if self.mask is not None:
            keep_prob = 1.0 - self.dropout_rate
            delta = delta * self.mask / keep_prob
        
        dZ = self.activation_function.derivative(self.z)
        delta *= dZ

        dW = np.dot(h.T, delta)
        dB = np.sum(delta, axis=0, keepdims=True)
        
        self.delta = np.dot(delta, self.W.T)
        return dW, dB




class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer, ...]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []
        delta = loss_grad
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]

            prev_act = input_data if i == 0 else self.layers[i-1].activations
            dW, dB = layer.backward(prev_act, delta)
            delta = layer.delta
            dl_dw_all.insert(0, dW)
            dl_db_all.insert(0, dB)

        return dl_dw_all, dl_db_all

    def train(
            self, 
            train_x: np.ndarray, 
            train_y: np.ndarray, 
            val_x: np.ndarray, 
            val_y: np.ndarray, 
            loss_func: LossFunction, 
            learning_rate: float=1E-3, 
            batch_size: int=16, 
            epochs: int=32,
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = []
        validation_losses = []
        
        for epoch in range(epochs):
            running_loss = 0
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                output = self.forward(batch_x, training=True)
                loss = loss_func.loss(batch_y, output)
                loss_grad = loss_func.derivative(batch_y, output)

                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)
                for i, (dl_dw, dl_db) in enumerate(zip(dl_dw_all, dl_db_all)):
                    self.layers[i].W -= learning_rate * dl_dw
                    self.layers[i].b -= learning_rate * dl_db
                
                running_loss += loss

            val_loss = 0
            for batch_x, batch_y in batch_generator(val_x, val_y, batch_size):
                output = self.forward(batch_x, training=False)
                loss = loss_func.loss(batch_y, output)
                val_loss += loss

            training_losses.append(running_loss / train_x.shape[0])
            validation_losses.append(val_loss / val_x.shape[0])

            print("-------------------------------------")
            print(f"Epoch: {epoch}, Train Loss: {running_loss}, Validation Loss: {val_loss}")
         
        return training_losses, validation_losses
