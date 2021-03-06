import numpy as np
from math import exp
from operator import add


class LogisticNeuron:
    def __init__(self, n_weights):
        self.weights = np.random.rand(n_weights, 1)
        self.total_error = 0
        self.cost = []
        self.u = np.array(0)
        self.error = 0

    def training(self, training_base, epochs, learning_rate):
        for i in range(epochs):
            np.random.shuffle(training_base)
            X = np.delete(training_base, -1, 1)
            Y = training_base[:, -1]

            predictions = self.predict(X)
            errors = Y - predictions

            activation_y = np.array(list(map(self.logistic_activation, self.u)))
            derived_function = np.array(list(map(lambda y: y*(1-y), activation_y)))

            x_y = np.array(list(map(np.dot, derived_function, X)))
            self.weights += np.array(learning_rate * np.dot(errors, x_y)).reshape((self.weights.shape[0], 1))

            cost = (errors ** 2).sum() / 2
            self.cost.append(cost)

            if cost == 0:
                break

    def training_net(self, x, learning_rate):
        def derivative_calc(y): return y * (1 - y)
        features = x[:-1]
        label = x[-1]
        prediction = self.predict(features)
        error = label - prediction
        y_calc = derivative_calc(self.logistic_activation(self.u))
        self.weights += np.array(learning_rate * error * y_calc * features).reshape((self.weights.shape[0], 1))
        self.total_error += error ** 2

    def calc_u(self, X):
        self.u = np.dot(X, self.weights)

    def logistic_activation(self, u):
        y = 1 / (1 + exp(-u))
        return y
        # return 1 if y >= 0.5 else 0

    def predict(self, X):
        self.calc_u(X)
        y_hat = np.array(list(map(self.logistic_activation, self.u)))

        for i in range(y_hat.shape[0]):
            y_hat[i] = 1 if y_hat[i] >= 0.5 else 0

        return y_hat

    def hit_rate(self, test_base):
        X = np.delete(test_base, -1, 1)
        Y = test_base[:, -1]

        y_hat = self.predict(X)

        hit = Y - y_hat
        hit = list(filter(lambda x: x == 0, hit))

        hit_rate = round((len(hit) / len(Y)) * 100, 2)

        print('Hit rate: {}%'.format(hit_rate))