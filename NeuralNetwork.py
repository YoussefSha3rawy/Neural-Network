from typing import List
from enum import Enum
import numpy as np
import numpy.typing as npt
from tabulate import tabulate
from HelperFunctions import LossFunctions, LossFunctionsDerivatives, LossFunctionEnum
from NeuralNetworkLayer import NeuralNetworkLayer, InputLayer, DenseLayer, DropoutLayer
from Optimizers import SGD, Optimizer
from sklearn import metrics
import os
import time


class NeuralNetworkPhase(Enum):
    TRAIN = 1
    TEST = 2


class NeuralNetwork:
    layers: List[NeuralNetworkLayer]
    loss_function: LossFunctionEnum
    phase: NeuralNetworkPhase
    optimizer: Optimizer

    def __init__(self, no_of_inputs: int):
        input_layer = InputLayer(no_of_inputs)
        self.layers = [input_layer]
        self.model_name = str(time.time())

    def add(self, layer: NeuralNetworkLayer):
        if isinstance(layer, DenseLayer):
            last_output = self.layers[0].no_of_inputs
            for l in reversed(self.layers):
                if (isinstance(l, DenseLayer)):
                    last_output = l.no_of_units
                    break
            layer.initialise_weights(last_output)

        self.layers.append(layer)

    def describe(self):
        tabulated_string = []
        for i, layer in enumerate(self.layers):
            if (isinstance(layer, DenseLayer)):
                tabulated_string.append(
                    [i, layer.__class__.__name__, layer.no_of_units, layer.weights.shape[0] * layer.weights.shape[1],
                     layer.bias.shape[1], layer.weights.shape[0] * layer.weights.shape[1] + layer.bias.shape[1]])
            elif isinstance(layer, InputLayer):
                tabulated_string.append(
                    [i, layer.__class__.__name__, layer.no_of_inputs, '', '', ''])
            else:
                tabulated_string.append(
                    [i, layer.__class__.__name__, '', '', '', ''])
        print(tabulate(tabulated_string, headers=[
              'Layer no', 'Layer type',  'No of units',
              'Weight Parameters', 'Bias Parameters', 'Total Parameters']))
        tabulated_string = np.array(tabulated_string)
        tabulated_string[tabulated_string == ''] = '0'
        print('Total Parameters:', np.sum(
            np.array(tabulated_string)[:, 5].astype(int)))

    def display_current_weights(self, display_bias=False, display_gradients=False):
        for i, layer in enumerate(self.layers):
            if (isinstance(layer, DenseLayer)):
                print('Layer', i+1, '\nWeights:',
                      layer.weights.shape, layer.weights, '\n')
                if display_gradients:
                    print(
                        f'Weight gradients: {layer.weight_gradients} {np.max(layer.weight_gradients)=}')
                if display_bias:
                    print('Bias:', layer.bias, '\n')

    def predict(self, X):
        self.phase = NeuralNetworkPhase.TEST
        self.forward_pass(X)
        return np.array(self.layers[-1].last_output)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike, epochs, optimizer: Optimizer, batch_size=None):
        self.optimizer = optimizer
        self.phase = NeuralNetworkPhase.TRAIN
        if batch_size is None:
            batch_size = 32
        reults = []
        if y.ndim == 1:
            y = y.reshape(1, -1)

        for epoch in range(epochs):
            start = time.time()
            print(f'Epoch {epoch+1}/{epochs}:')
            y_preds = []
            for i in range(0, X.shape[0], batch_size):
                print(
                    f'Step {int(i / batch_size)} / {int(X.shape[0] / batch_size)}', end='\r')
                X_batch = X[i: np.minimum(i + batch_size, X.shape[0])]
                y_batch = y[i: np.minimum(i + batch_size, X.shape[0])]
                self.forward_pass(X_batch)
                y_preds.extend(self.layers[-1].last_output)
                self.backward_pass(y_batch)
                self.update_params()
            y_preds = np.array(y_preds)
            epoch_results = self.calculate_loss(y, y_preds)
            epoch_results['time_per_epoch'] = f'{time.time() - start:.2f}'
            epoch_results['learning_rate'] = f'{self.optimizer.learning_rate:.4f}'
            print(epoch_results)
            reults.append(epoch_results)
            self.optimizer.update_learning_rate()
        return reults

    def update_params(self):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                self.optimizer.update_params(layer)

    def calculate_loss(self, y, y_preds):
        pred = np.argmax(y_preds, axis=1)
        y_compare = np.argmax(y, axis=1)
        score = metrics.accuracy_score(y_compare, pred)

        loss = LossFunctions.cross_entropy(y, y_preds)
        if self.optimizer.l1_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    loss += np.sum(np.abs(layer.weights)) * \
                        self.optimizer.l1_lambda
        if self.optimizer.l2_lambda > 0:
            for layer in self.layers:
                if isinstance(layer, DenseLayer):
                    loss += np.sum(np.square(layer.weights)) * \
                        self.optimizer.l2_lambda
        return {'loss': f'{loss:.3f}', 'accuracy': f'{score:.3f}'}

    def forward_pass(self, X):
        for layer in self.layers:
            if not isinstance(layer, DropoutLayer) or self.phase == NeuralNetworkPhase.TRAIN:
                layer.forward(X)
                X = layer.last_output

    def backward_pass(self, y_actual):
        output_final_layer = self.layers[-1].last_output
        delta = LossFunctionsDerivatives.cross_entropy(
            output_final_layer, y_actual, self.layers[-1].__class__.__name__)
        for layer in reversed(self.layers[1:]):
            layer.backward(delta)
            delta = layer.last_delta

    def save_model(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        model_directory = os.path.join(curr_path, 'models', self.model_name)
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                np.save(os.path.join(model_directory,
                        f'weights_{i}'), layer.weights)
                np.save(os.path.join(model_directory,
                        f'bias_{i}'), layer.bias)

    def load_weights(self):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        model_directory = os.path.join(curr_path, 'models', self.model_name)
        for i, layer in enumerate(self.layers):
            if isinstance(layer, DenseLayer):
                layer.weights = np.load(os.path.join(model_directory,
                                                     f'weights_{i}.npy'))
                layer.bias = np.load(os.path.join(model_directory,
                                                  f'bias_{i}.npy'))
