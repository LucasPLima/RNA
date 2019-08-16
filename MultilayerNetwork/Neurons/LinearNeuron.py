import numpy as np


class LinearNeuronML:
    def __init__(self, n_weights):
        self.x = []
        self.u = 0
        self.d_y = 1
        self.output = 0
        self.weights = np.random.random(n_weights)

    def process_inputs(self, x):
        self.x = x
        self.u = np.dot(self.weights.T, x)
        self.output = self.u

    def adjust_weights(self, error, learning_rate):
        self.weights = self.weights + np.dot((learning_rate * error * self.d_y), self.x)

    def predict(self):
        return self.output
