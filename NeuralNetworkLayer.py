import numpy as np
import numpy.typing as npt
from HelperFunctions import ActivationFunctions


class NeuralNetworkLayer:
    last_input: npt.ArrayLike
    last_output: npt.ArrayLike
    last_delta: npt.ArrayLike

    def forward(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        self.last_input = X
        self.last_output = X
        return X

    def backward(self, delta_next_layer):
        return delta_next_layer


class InputLayer(NeuralNetworkLayer):
    no_of_inputs: int

    def __init__(self, no_of_inputs):
        self.no_of_inputs = no_of_inputs


class DenseLayer(NeuralNetworkLayer):
    no_of_units: int
    weights: npt.ArrayLike
    bias: npt.ArrayLike
    weight_gradients: npt.ArrayLike
    bias_gradients: npt.ArrayLike

    def __init__(self, no_of_units: int):
        self.no_of_units = no_of_units
        self.bias = np.zeros((1, no_of_units))

    def initialise_weights(self, last_layer_no_outputs: int):
        limit = np.sqrt(6 / (last_layer_no_outputs + self.no_of_units))
        self.weights = np.random.uniform(-limit, limit,
                                         size=(last_layer_no_outputs, self.no_of_units))

    def forward(self, X):
        self.last_input = X
        self.last_output = np.dot(X, self.weights) + self.bias

    def backward(self, delta_next_layer):
        self.weight_gradients = np.dot(
            self.last_input.T, delta_next_layer)
        self.bias_gradients = np.sum(delta_next_layer, axis=0, keepdims=True)
        self.last_delta = np.dot(delta_next_layer, self.weights.T)


class DropoutLayer(NeuralNetworkLayer):
    dropout_type: str
    dropout_prob: float
    dropout_mask: npt.ArrayLike
    inverted_dropout: bool

    def __init__(self, dropout_prob: float = 0.2):
        self.dropout_prob = dropout_prob

    def forward(self, X):
        self.last_input = X
        self.dropout_mask = (np.random.rand(
            X.shape[0], 1) > self.dropout_prob) / self.dropout_prob
        self.last_output = self.dropout_mask * X

    def backward(self, delta_next_layer):
        self.last_delta = (delta_next_layer *
                           self.dropout_mask) / self.dropout_prob


class RelUActivation(NeuralNetworkLayer):
    def forward(self, X):
        self.last_input = X
        self.last_output = ActivationFunctions.relU_activate(X)

    def backward(self, delta_next_layer):
        self.last_delta = delta_next_layer.copy()
        self.last_delta *= ActivationFunctions.relU_derivative(
            self.last_output)


class SigmoidActivation(NeuralNetworkLayer):
    def forward(self, X):
        self.last_input = X
        self.last_output = ActivationFunctions.sigmoid_activate(X)

    def backward(self, delta_next_layer):
        self.last_delta = delta_next_layer.copy()
        self.last_delta *= ActivationFunctions.sigmoid_derivative(
            self.last_output)


class SoftmaxActivation(NeuralNetworkLayer):
    def forward(self, X):
        self.last_input = X
        self.last_output = ActivationFunctions.softmax_activate(X)

    def backward(self, delta_next_layer):
        # # self.last_delta = delta_next_layer * \
        # #     ActivationFunctions.softmax_derivative(self.last_output[-1])

        # jacobian_matrix = np.diagflat(self.last_output[-1]) - \
        #     np.dot(self.last_output[-1], self.last_output[-1].T)
        # # Calculate sample-wise gradient
        # # and add it to the array of sample gradients
        # self.last_delta = np.dot(delta_next_layer, jacobian_matrix)

        self.last_delta = delta_next_layer
