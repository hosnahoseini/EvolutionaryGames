import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # TODO
        # layer_sizes example: [4, 10, 2]
        self.W1 = np.random.normal(size=(layer_sizes[1], layer_sizes[0]))
        self.W2 = np.random.normal(size=(layer_sizes[2], layer_sizes[1]))

        self.b1 = np.zeros((layer_sizes[1], 1))
        self.b2 = np.zeros((layer_sizes[2], 1))

    def activation(self, x):
        
        # TODO
        return 1 / (1 + np.exp(x))

    def forward(self, x):
        # TODO
        # x example: np.array([[0.1], [0.2], [0.3]])
        A1 = self.W1 @ x + self.b1
        Z1 = self.activation(A1)
        A2 = self.W2 @ Z1 + self.b2
        Z2 = self.activation(A2)

        return Z2