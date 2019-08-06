import numpy as np
from math import exp


class LogisticNeuronML:
    def __init__(self, n_weights):
        self.x = []
        self.u = 0
        self.d_y = 0
        self.output = 0
        self.weights = np.random.random(n_weights + 1)

    def process_inputs(self, x):
        self.x = x
        self.u = np.dot(self.weights, x)
        self.d_y = self.u * (1 - self.u)
        self.classify()

    def classify(self):
        activation = 1 / (1 + exp(self.u))
        self.output = 1 if activation >= 0.5 else 0

    def adjust_weights(self, error, learning_rate):
        self.weights = self.weights + np.dot((learning_rate * error * self.d_y), self.x)

