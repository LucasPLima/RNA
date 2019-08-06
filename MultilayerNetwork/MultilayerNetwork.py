import numpy as np
import copy
from MultilayerNetwork.Layer import Layer


class MLP:
    def __init__(self, n_layers, n_neurons, n_features, hidden_layer_func='Logistic'):
        self.n_layers = n_layers
        self.n_features = n_features
        self.layers = [Layer(is_output_layer=False, nu_neurons=n_neurons[0], nu_features=self.n_features)]
        for i in range(1, n_layers):
            if i != n_layers-1:
                self.layers.append(Layer(is_output_layer=False, nu_neurons=n_neurons[i], nu_features=n_neurons[i-1]))
            else:
                self.layers.append(Layer(is_output_layer=True, nu_neurons=n_neurons[i], nu_features=n_neurons[i-1]))

    def training(self, training_base, epochs, learning_rate):
        for i in range(epochs):
            np.random.shuffle(training_base)
            for sample in training_base:
                x_training = sample[:self.n_features]
                y_training = sample[self.n_features:]
                self.forward_propagation(x_training)
                self.back_propagation_adjust(y_training, learning_rate)

    def forward_propagation(self, input_data):
        new_input = input_data
        for layer in self.layers:
            output_data = layer.forward_process(new_input)
            new_input = output_data

    def back_propagation_adjust(self, desired_output, learning_rate):
        layers_r = copy.deepcopy(self.layers)
        layers_r.reverse()
        for layer in layers_r:
            if layer.is_output_layer:
                layer.calc_layer_error(desired_output)
            else:
                # TODO
                # Lembrar que a lista de layers é reversa, layer_forward deve ser (layer atual - 1)
                pass
            layer.adjust_neurons(learning_rate)

