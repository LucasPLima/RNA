import numpy as np
from math import exp


class LogisticNeuronML:
    def __init__(self, n_weights):
        self.x = []
        self.u = 0
        self.d_y = 0
        self.output = 0
        self.weights = np.random.random(n_weights + 1)

    # TODO
    # Lembrar da saída / erro pertence aos reais/ não classificar
    def process_inputs(self, x):
        self.x = x
        self.u = np.dot(self.weights.T, x)
        self.output = 1 / (1 + exp(self.u))
        self.d_y = self.output * (1 - self.output)

    def adjust_weights(self, error, learning_rate):
        self.weights = self.weights + np.dot((learning_rate * error * self.d_y), self.x)

    # TODO
    # Não utilizar durante o treino
    def predict(self):
        pass
