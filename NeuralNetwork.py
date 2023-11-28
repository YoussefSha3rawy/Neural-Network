from typing import List, Any
from enum import Enum
import numpy as np
from tabulate import tabulate


class ActivationFunctionEnum(Enum):
    RELU = 'RELU'
    SIGMOID = 'SIGMOID'
    SOFTMAX = 'SOFTMAX'


class ActivationFunctions:
    def activate(X, activation_function: ActivationFunctionEnum):
        if activation_function == ActivationFunctionEnum.RELU:
            return ActivationFunctions.relU_activate(X)
        elif activation_function == ActivationFunctionEnum.SOFTMAX:
            return ActivationFunctions.softmax_activate(X)
        elif activation_function == ActivationFunctionEnum.SIGMOID:
            return ActivationFunctions.sigmoid_activate(X)

    def relU_activate(X):
        return np.maximum(X, 0)

    def sigmoid_activate(X):
        return 1 / (1 + np.exp(-X))

    def softmax_activate(X):
        return np.exp(X) / np.exp(X).sum()


class NeuralNetworkLayer:
    no_of_units: int
    activation_function: ActivationFunctionEnum
    weights: List[Any]
    bias: List[Any]
    dropout_type: str
    dropout_prob: float

    def __init__(self, no_of_units: int,
                 activation_function: ActivationFunctionEnum,
                 weights, dropout_type: str = None, dropout_prob: float = 0.2):
        self.no_of_units = no_of_units
        self.activation_function = activation_function
        self.weights = weights
        if weights is not None:
            self.bias = np.random.standard_normal((1, no_of_units))
            self.dropout_type = dropout_type
            self.dropout_prob = dropout_prob


class NeuralNetwork:
    layers: List[NeuralNetworkLayer]

    def __init__(self, no_of_inputs: int):
        input_layer = NeuralNetworkLayer(no_of_inputs, None, None)
        self.layers = [input_layer]

    def add_hidden_layer(self, no_of_units: int,
                         activation_function: ActivationFunctionEnum,
                         dropout_type: str = None, dropout_prob: float = 0.2):
        self.layers.append(NeuralNetworkLayer(
            no_of_units, activation_function, np.random.standard_normal((
                self.layers[-1].no_of_units, no_of_units)), dropout_type,
            dropout_prob))

    def describe(self):
        tabulated_string = [[i, layer.no_of_units,
                             layer.activation_function.value,
                             layer.weights.shape[0] * layer.weights.shape[1],
                             layer.bias.shape[1],
                             layer.weights.shape[0] * layer.weights.shape[1] +
                             layer.bias.shape[1]]
                            for i, layer in enumerate(self.layers)
                            if layer.weights is not None]
        tabulated_string.insert(0, [0, self.layers[0].no_of_units,
                                    None, 0, 0, 0])
        print(tabulate(tabulated_string, headers=[
              'Layer no', 'No of units', 'Activation Function',
              'Weight Parameters', 'Bias Parameters', 'Total Parameters']))
        print('Total Parameters:', np.sum(np.array(tabulated_string)[:, 5]))

    def predict(self, x):
        return self.forward_pass(x)

    def forward_pass(self, x):
        last_layer_output = x
        for i, layer in enumerate(self.layers[1:]):
            last_layer_output = np.dot(last_layer_output, layer.weights)
            last_layer_output = last_layer_output + layer.bias
            last_layer_output = ActivationFunctions.activate(
                last_layer_output, layer.activation_function)
            if layer.dropout_type is not None:
                dp = np.random.rand(*last_layer_output.shape)
                last_layer_output = (
                    dp > layer.dropout_prob) * last_layer_output
        return last_layer_output
