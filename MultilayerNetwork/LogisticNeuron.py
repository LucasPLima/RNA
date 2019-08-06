import numpy as np
from math import exp


class LogisticNeuronML:
    def __init__(self, n_weights):
        self.u = 0
        self.d_u = 0
        self.output = 0
        self.weights = np.random.random(n_weights + 1)

    def process_inputs(self, x):
        self.u = np.dot(self.weights, x)
        self.d_u = self.u * (1 - self.u)
        self.classify()

    def classify(self):
        activation = 1 / (1 + exp(self.u))
        self.output = 1 if activation >= 0.5 else 0

    def adjust_weights(self):
        pass
