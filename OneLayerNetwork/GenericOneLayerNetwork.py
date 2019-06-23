import numpy as np
import SimplePerceptron.NeuronioPerceptron as perceptron
import GenericNeuron.HiperbolicNeuron as hiperbolic
import GenericNeuron.LogisticNeuron as logistic


class GenericOLN:
    def __init__(self, n_labels, n_weights, neuron_type='P'):
        if neuron_type == 'P':
            self.neurons = [perceptron.NeuronioMP(n_weights) for i in range(n_labels)]
        elif neuron_type == 'H':
            self.neurons = [hiperbolic.HiperbolicNeuron(n_weights) for i in range(n_labels)]
        elif neuron_type == 'L':
            self.neurons = [logistic.LogisticNeuron(n_weights) for i in range(n_labels)]
        self.u = []
        self.neuron_type = neuron_type
        self.weights = [[] for i in range(n_labels)]
        self.epochs_error = []

    def training(self, epochs, learning_rate, training_base):
        n_labels = len(self.neurons)
        for epoch in range(epochs):
            error = 0
            np.random.shuffle(training_base)
            for n in range(n_labels):
                label_choiced = (n_labels - n) * (-1)
                X_training = training_base[:, 0:(n_labels * -1)]
                Y_training = training_base[:, label_choiced]
                new_training_base = np.concatenate((X_training, Y_training[:, None]), axis=1)
                self.neurons[n].training(new_training_base, epochs=1, learning_rate=learning_rate)
                error += self.neurons[n].cost[epoch]
                self.weights[n] = np.array(self.neurons[n].weights)
            self.epochs_error.append(error)

            if error == 0:
                return

    def predict(self, x):
        y_predict = []
        self.u = []
        for neuron in self.neurons:
            y_predict.append(neuron.predict(x))
            self.u.append(neuron.u)
            np.array(y_predict)

        if self.neuron_type != 'P':
            y_predict = [i[0] for i in y_predict]

        if y_predict.count(1) > 1:
            choiced_label = self.u.index(max(self.u))
            y_predict = np.zeros(len(self.neurons))
            y_predict[choiced_label] = 1

        return y_predict

    def hit_rate(self, test_base):
        labels = len(self.neurons)
        X_test = test_base[:, 0:(-labels)]
        Y_test = test_base[:, -labels:]

        y_predict = np.array([self.predict(x) for x in X_test])
        test = Y_test - y_predict
        hit = list(filter(lambda x: sum(x) == 0, Y_test - y_predict))

        hit_rate = (len(hit)/Y_test.shape[0]) * 100

        return hit_rate, y_predict
