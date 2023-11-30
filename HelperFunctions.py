from enum import Enum
import numpy as np
import numpy.typing as npt


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
        else:
            return X

    def activation_derivitive(a, activation_function: ActivationFunctionEnum):
        if activation_function == ActivationFunctionEnum.RELU:
            return ActivationFunctions.relU_derivative(a)
        elif activation_function == ActivationFunctionEnum.SOFTMAX:
            return ActivationFunctions.softmax_derivative(a)
        elif activation_function == ActivationFunctionEnum.SIGMOID:
            return ActivationFunctions.sigmoid_derivative(a)
        else:
            return np.ones(a.shape)

    def relU_activate(X):
        return np.maximum(X, 0)

    def sigmoid_activate(X):
        return 1 / (1 + np.exp(-X))

    def softmax_activate(X):
        return np.exp(X) / np.exp(X).sum()

    def sigmoid_derivative(a):
        return a * (1-a)

    def relU_derivative(a):
        return (a >= 0) * 1

    def softmax_derivative(a):
        pass


class LossFunctions:
    def squared_error(y_actual: npt.ArrayLike, y_pred: npt.ArrayLike) -> npt.ArrayLike:
        return np.square(y_pred - y_actual)

    def mean_squared_error(y_actual: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
        return np.mean(np.square(y_pred - y_actual))

    def cross_entropy(y_actual: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
        N = y_pred.shape[0]
        ce = -np.sum(y_actual*np.log(y_pred))/N
        return ce


class LossFunctionsDerivatives:
    def squared_error_derivative(a, y):
        return y - a
