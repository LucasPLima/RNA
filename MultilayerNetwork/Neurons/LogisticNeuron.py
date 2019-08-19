import numpy as np
from math import exp


class LogisticNeuronML:
    def __init__(self, n_weights):
        self.x = []
        self.u = 0
        self.d_y = 0
        self.output = 0
        self.weights = np.random.random(n_weights)

    def process_inputs(self, x):
        self.x = x
        self.u = np.dot(self.weights.T, x)
        self.output = 1 / (1 + exp(-self.u))
        self.d_y = self.output * (1 - self.output)

    @staticmethod
    def act_function(u):
        return 1 / (1+np.exp(-u))

    def adjust_weights(self, error, learning_rate):
        self.weights = self.weights + np.dot((learning_rate * error * self.d_y), self.x)

    def predict(self):
        if self.output >= 0.5:
            return 1
        else:
            return 0
