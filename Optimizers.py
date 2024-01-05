from NeuralNetworkLayer import DenseLayer
import numpy as np


class Optimizer:
    learning_rate: float

    def update_learning_rate(self):
        pass

    def update_params(self, layers: list[DenseLayer]):
        pass


class SGD:
    l1_lambda: float
    l2_lambda: float
    momentum = float
    decay: float
    cycles: int

    def __init__(self, learning_rate=0.01, momentum=0.0, decay=0.0, l1_lambda=0.0, l2_lambda=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.decay = decay
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.cycles = 0

    def update_learning_rate(self):
        if self.decay > 0:
            self.learning_rate = self.learning_rate * \
                (1 / (1 + self.decay * self.cycles))
        self.cycles += 1

    def update_params(self, layer: DenseLayer):
        if self.l1_lambda > 0:
            delta_weights_L1 = np.where(layer.weights < 0, -1, 1)
            layer.weight_gradients += self.l1_lambda * delta_weights_L1

            delta_L1_bias = np.where(layer.bias < 0, -1, 1)
            layer.bias_gradients += self.l1_lambda * delta_L1_bias

        if self.l2_lambda > 0:
            layer.weight_gradients += 2 * self.l2_lambda * layer.weights
            layer.bias_gradients += 2 * self.l2_lambda * layer.bias

        if self.momentum > 0:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.bias)

            weight_updates = self.momentum * layer.weight_momentums - \
                self.learning_rate * layer.weight_gradients
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - \
                self.learning_rate * layer.bias_gradients
            layer.bias_momentums = bias_updates
        else:
            weight_updates = - self.learning_rate * layer.weight_gradients
            bias_updates = - self.learning_rate * layer.bias_gradients

        layer.weights += weight_updates
        layer.bias += bias_updates
