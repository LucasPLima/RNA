import SimplePerceptron.NeuronioPerceptron as perceptron


class OLPerceptron:
    def __init__(self, n_labels, n_weights):
        self.neurons = [perceptron.NeuronioMP(n_weights) for i in range(n_labels)]

    def training(self, epochs, learning_rate, training_base):
        exit()

    def predict(self, X):
        exit()

    def hit_rate(self, test_base):
        exit()