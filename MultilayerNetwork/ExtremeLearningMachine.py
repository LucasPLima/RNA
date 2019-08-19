import numpy as np
from Utils.model_utils import validate_predict
from .Layer import Layer
from .Neurons import LogisticNeuron as lN


class ELM:
    def __init__(self, n_layers,
                 n_neurons,
                 n_features,
                 output_layer_func='Logistic'):
        self.n_layers = n_layers
        self.n_features = n_features
        self.layers = [Layer(is_output_layer=False, nu_neurons=n_neurons[0], nu_features=self.n_features)]
        for i in range(1, n_layers):
            if i != n_layers - 1:
                self.layers.append(
                    Layer(is_output_layer=False, nu_neurons=n_neurons[i], nu_features=n_neurons[i - 1]))
            else:
                self.layers.append(
                    Layer(is_output_layer=True, nu_neurons=n_neurons[i], nu_features=n_neurons[i - 1],
                          neuron_type=output_layer_func))

    def training(self, training_base):
        np.random.shuffle(training_base)
        X_training = training_base[:, :self.n_features + 1]
        Y_training = training_base[:, self.n_features + 1:]
        self.feed_forward(X_training, Y_training)

    def get_weights(self):
        H = []
        O = []

        for layer in self.layers:
            if not layer.is_output_layer:
                for neuron in layer.neurons:
                    H.append(neuron.weights)
            else:
                for neuron in layer.neurons:
                    O.append(neuron.weights)

        return np.array(H), np.array(O)

    def feed_forward(self, X_training, Y_training):
        H, O = self.get_weights()
        U = X_training.dot(H.T)

        ACT_U = lN.LogisticNeuronML.act_function(np.array(U, dtype=np.float32))
        ACT_U = np.append(-np.ones(ACT_U.shape[0]).reshape((ACT_U.shape[0], 1)), ACT_U, axis=1)
        ACT_U_T = np.transpose(ACT_U)

        M = np.dot(np.linalg.pinv(np.dot(ACT_U_T, ACT_U)), np.dot(ACT_U_T, Y_training))

        for i in range(len(self.layers[-1].neurons)):
            self.layers[-1].neurons[i].weights = M.T[i]

    def forward_propagation(self, input_data):
        new_input = input_data
        for layer in self.layers:
            output_data = layer.forward_process(new_input)
            new_input = output_data

    def predict(self, test_base, classify=True):
        predicted_labels = []
        neuron_activations = []
        for sample in test_base:
            x_test = sample[:self.n_features + 1]
            self.forward_propagation(x_test)
            predict = [neuron.predict() for neuron in self.layers[-1].neurons]
            u_s = [neuron.u for neuron in self.layers[-1].neurons]
            neuron_activations.append(u_s)
            predicted_labels.append(predict)

        if classify:
            for i in range(len(predicted_labels)):
                predicted_labels[i] = validate_predict(predicted_labels[i], neuron_activations[i])

        return predicted_labels, neuron_activations
