from enum import Enum
import numpy as np
import numpy.typing as npt
import sys


class LossFunctionEnum(Enum):
    MSE = 'MSE'
    CCE = 'CCE'


class ActivationFunctionEnum(Enum):
    RELU = 'RELU'
    SIGMOID = 'SIGMOID'
    SOFTMAX = 'SOFTMAX'
    NONE = 'NONE'


class ActivationFunctions:
    def activate(X, activation_function: ActivationFunctionEnum):
        if activation_function == ActivationFunctionEnum.RELU:
            return ActivationFunctions.relU_activate(X)
        elif activation_function == ActivationFunctionEnum.SOFTMAX:
            return ActivationFunctions.softmax_activate(X)
        elif activation_function == ActivationFunctionEnum.SIGMOID:
            return ActivationFunctions.sigmoid_activate(X)
        elif activation_function == ActivationFunctionEnum.NONE:
            return X

    def activation_derivitive(a, activation_function: ActivationFunctionEnum):
        if activation_function == ActivationFunctionEnum.RELU:
            return ActivationFunctions.relU_derivative(a)
        elif activation_function == ActivationFunctionEnum.SOFTMAX:
            return ActivationFunctions.softmax_derivative(a)
        elif activation_function == ActivationFunctionEnum.SIGMOID:
            return ActivationFunctions.sigmoid_derivative(a)
        elif activation_function == ActivationFunctionEnum.NONE:
            return np.ones(a.shape)

    def relU_activate(X):
        return np.maximum(X, 0)

    def sigmoid_activate(X):
        return 1 / (1 + np.exp(-X))

    def softmax_activate(X):
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_values / exp_values.sum(axis=1, keepdims=True)

    def sigmoid_derivative(a):
        return a * (1-a)

    def relU_derivative(a):
        return (a > 0) * 1

    def softmax_derivative(a):
        I = np.eye(a.shape[-1])
        return np.dot((I - a.T), a)


class LossFunctions:
    def squared_error(y_actual: npt.ArrayLike, y_pred: npt.ArrayLike) -> npt.ArrayLike:
        return np.square(y_pred - y_actual)

    def mean_squared_error(y_actual: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
        return np.mean(np.square(y_pred - y_actual))

    def cross_entropy(targets, predictions, epsilon=1e-12):
        predictions = np.clip(predictions, epsilon, 1. - epsilon)
        ce = -np.mean(targets*np.log(predictions+1e-9))
        return ce


class LossFunctionsDerivatives:
    def squared_error_derivative(y_pred, y_actual):
        return y_pred - y_actual

    def cross_entropy(y_pred, y_actual, final_layer_name):
        if final_layer_name == 'SoftmaxActivation':
            return (y_pred - y_actual) / len(y_pred)
