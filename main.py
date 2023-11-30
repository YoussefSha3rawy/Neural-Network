import numpy as np
from HelperFunctions import LossFunctions
from NeuralNetwork import NeuralNetwork, ActivationFunctionEnum, ActivationFunctions
import matplotlib.pyplot as plt

nn = NeuralNetwork(3)
nn.add_hidden_layer(16, ActivationFunctionEnum.SIGMOID)
nn.add_hidden_layer(32, ActivationFunctionEnum.SIGMOID)
nn.add_hidden_layer(2, ActivationFunctionEnum.SOFTMAX)

nn.describe()

X = np.random.rand(3, 1)
y = [[1], [0]]
prediction = nn.predict(X)

print(np.argmax(prediction))


epochs = 1000
lr = 0.001
losses = nn.train(X, y, epochs, lr)
plt.scatter(list(range(1, epochs+1)), losses)
plt.show()
