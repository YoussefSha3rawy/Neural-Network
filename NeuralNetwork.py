from typing import List, Any
import numpy as np
import numpy.typing as npt
from tabulate import tabulate
from HelperFunctions import ActivationFunctionEnum, ActivationFunctions, LossFunctions, LossFunctionsDerivatives


class NeuralNetworkLayer:
    no_of_units: int
    activation_function: ActivationFunctionEnum
    weights: npt.ArrayLike
    bias: npt.ArrayLike
    last_output: npt.ArrayLike
    last_delta: npt.ArrayLike
    dropout_type: str
    dropout_prob: float

    def __init__(self, no_of_units: int,
                 activation_function: ActivationFunctionEnum,
                 weights, dropout_type: str = None, dropout_prob: float = 0.2):
        self.no_of_units = no_of_units
        self.activation_function = activation_function
        self.weights = weights
        if weights is not None:
            self.bias = np.random.standard_normal((no_of_units, 1))
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

    def describe(self, include_input_layer=False):
        tabulated_string = [[i, layer.no_of_units,
                             layer.activation_function.value,
                             layer.weights.shape[0] * layer.weights.shape[1],
                             layer.bias.shape[0],
                             layer.weights.shape[0] * layer.weights.shape[1] +
                             layer.bias.shape[0]]
                            for i, layer in enumerate(self.layers)
                            if layer.weights is not None]
        if include_input_layer:
            tabulated_string.insert(0, [0, self.layers[0].no_of_units,
                                    None, 0, 0, 0])
        print(tabulate(tabulated_string, headers=[
              'Layer no', 'No of units', 'Activation Function',
              'Weight Parameters', 'Bias Parameters', 'Total Parameters']))
        print('Total Parameters:', np.sum(
            np.array(tabulated_string)[:, 5].astype(int)))

    def display_current_weights(self):
        for i, layer in enumerate(self.layers[1:]):
            print('Layer', i+1, '\n', layer.weights, '\n', layer.bias)
            print('\n')

    def predict(self, X):
        return self.forward_pass(X)

    def train(self, X, y, epochs, learning_rate=0.01):
        losses = []
        for i in range(epochs):
            y_pred = self.forward_pass(X)
            self.backward_pass(y, learning_rate)
            print(f'Epoch {i+1}/{epochs}:')
            loss = LossFunctions.mean_squared_error(y, y_pred)
            losses.append(loss)
            print(f'Loss:{loss}')
            print(y_pred)
        return losses

    def forward_pass(self, X):
        self.layers[0].last_output = X
        last_layer_output = X
        for i, layer in enumerate(self.layers[1:]):
            last_layer_output = np.dot(layer.weights.T, last_layer_output)
            last_layer_output = last_layer_output + layer.bias
            last_layer_output = ActivationFunctions.activate(
                last_layer_output, layer.activation_function)
            if layer.dropout_type is not None:
                dp = np.random.rand(*last_layer_output.shape)
                last_layer_output = (
                    dp > layer.dropout_prob) * last_layer_output
            self.layers[i+1].last_output = last_layer_output
        return last_layer_output

    def backward_pass(self, y_actual, learning_rate):
        output_final_layer = self.layers[-1].last_output
        delta_final_layer = (y_actual - output_final_layer) * \
            (output_final_layer * (1 - output_final_layer))
        self.layers[-1].last_delta = delta_final_layer
        new_weight_final_layer = self.layers[-1].weights + np.dot(self.layers[-2].last_output,
                                                                  delta_final_layer.T) * learning_rate
        new_weights = [new_weight_final_layer]
        for i in reversed(range(1, len(self.layers) - 1)):
            delta_next_layer = self.layers[i+1].last_delta
            weight_next_layer = self.layers[i+1].weights
            delta_current_layer = np.dot(weight_next_layer, delta_next_layer) * (
                self.layers[i].last_output * (1 - self.layers[i].last_output))
            self.layers[i].last_delta = delta_current_layer
            new_weight = self.layers[i].weights + np.dot(
                self.layers[i-1].last_output, delta_current_layer.T) * learning_rate
            new_weights.insert(0, new_weight)

        for i, layer in enumerate(self.layers[1:]):
            layer.weights = new_weights[i]
