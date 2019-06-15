import numpy as np
import SimplePerceptron.NeuronioPerceptron as perceptron


class OLPerceptron:
    def __init__(self, n_labels, n_weights):
        self.neurons = [perceptron.NeuronioMP(n_weights) for i in range(n_labels)]
        self.u = np.array(len(self.neurons))

    def training(self, epochs, learning_rate, training_base):
        n_labels = len(self.neurons)
        for i in range(epochs):
            np.random.shuffle(training_base)
            for n in range(n_labels):
                label_choiced = (n_labels - n) * (-1)
                Y_training= training_base[:, label_choiced]
                X_training= training_base[:, 0:(n_labels * -1) - 1]
                new_training_base = np.append(X_training, Y_training, axis=1)
                self.neurons[n].training(new_training_base, epochs=1, learning_rate=learning_rate)

    def predict(self, x):
        exit()

    def hit_rate(self, test_base):
        exit()