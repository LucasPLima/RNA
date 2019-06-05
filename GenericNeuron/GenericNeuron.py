import numpy as np
from operator import add
from math import exp


class GenericNeuron:
    #type 'L' for Logistic and 'H'  for Hyperbolic
    def __init__(self, n_weights, type):
        self.weights = np.random.rand(n_weights, 1)
        self.neuron_type = type
        self.cost = []
        self.u = 0

    def training(self, training_base, n_epochs, learning_rate):
        for i in range(n_epochs):
            np.random.shuffle(training_base)
            X = np.delete(training_base, -1, 1)
            Y = training_base[:, -1]
            predictions = self.predict(X)
            errors = Y - predictions
            u = np.dot(X, self.weights)
            if self.neuron_type == 'L':
                activation_y = list(map(self.logistic_activation, u))
                derived_function = list(map(lambda y: y*(1-y), activation_y))
            else:
                activation_y = list(map(self.hiperbolic_activation, u))
                derived_function = list(map(lambda y: 0.5 * (1 - y**2), activation_y))
            x_y = np.array(list(map(np.dot, derived_function, X)))
            atual = (learning_rate * np.dot(x_y.T, errors))
            self.weights = list(map(add, self.weights, atual.astype(float)))
            cost = (errors ** 2).sum() / 2
            self.cost.append(cost)

            if cost == 0:
                break

    def logistic_activation(self, u):
        y = 1 / (1 + exp(-u))
        return y
        # return 1 if y >= 0.5 else 0

    def hiperbolic_activation(self, u):
        y = (1-exp(-u))/(1+exp(-u))
        return y
        # return 1 if y >= 0 else -1

    def predict(self, X):
        self.u = np.dot(X, self.weights)

        if self.neuron_type == 'L':
            y_hat = list(map(self.logistic_activation, self.u))
            for i in range(len(y_hat)):
                y_hat[i] = 1 if y_hat[i] >= 0.5 else 0
        else:
            y_hat = list(map(self.hiperbolic_activation, self.u))
            for i in range(len(y_hat)):
                y_hat[i] = 1 if y_hat[i] >= 0 else -1

        return y_hat

    def hit_rate(self, test_base):
        X = np.delete(test_base, -1, 1)
        Y = test_base[:, -1]

        y_hat = self.predict(X)

        hit = Y - y_hat
        hit = list(filter(lambda x: x == 0, hit))

        hit_rate = round((len(hit) / len(Y)) * 100, 2)

        print('Hit rate: {}%'.format(hit_rate))