import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt


class Adaline:
    def __init__(self, n_weights=1):
        self.nweights = n_weights
        self.weights = np.random.rand(self.nweights)
        self.cost = []

    def training(self, training_base, epochs, learning_rate):
        for i in range(epochs):
            np.random.shuffle(training_base)
            X = np.delete(training_base, -1, 1)
            y = training_base[:, -1]
            predictions = self.activation(X)
            errors = y - predictions
            self.weights += learning_rate * np.dot(X.T, errors)
            cost = (errors**2).sum() / 2
            self.cost.append(cost)

            if cost == 0:
                print('The error was reached to zero.')
                break

    def activation(self, X):
        return np.dot(self.weights, X.T)

    def test(self, test_base):
        X_training = np.delete(test_base, -1, 1)
        Y_training = test_base[:, -1]

        prediction = self.activation(X_training)

        mean_se = round(mean_squared_error(Y_training, prediction), 4)
        rad_mse = sqrt(mean_se)

        print('MSE:{}'.format(mean_se))
        print('RMSE:{}'.format(rad_mse))

        return mean_se, rad_mse