import numpy as np
from NeuralNetwork import NeuralNetwork, ActivationFunctionEnum

nn = NeuralNetwork(3)
nn.add_hidden_layer(8, ActivationFunctionEnum.RELU)
nn.add_hidden_layer(16, ActivationFunctionEnum.SIGMOID, '', 0.1)
nn.add_hidden_layer(8, ActivationFunctionEnum.SOFTMAX)

nn.describe()

prediction = nn.predict(np.random.rand(1, 3))

print(np.argmax(prediction))
