from MultilayerNetwork.Layer import Layer


class MLP:
    def __init__(self, n_layers, n_neurons, n_features, hidden_layer_func='Logistic'):
        self.n_layers = n_layers
        self.layers = [Layer(is_output_layer=False, nu_neurons=n_neurons[0], nu_features=n_features)]
        for i in range(1, n_layers):
            if i != n_layers-1:
                self.layers.append(Layer(is_output_layer=False, nu_neurons=n_neurons[i], nu_features=n_neurons[i-1]))
            else:
                self.layers.append(Layer(is_output_layer=True, nu_neurons=n_neurons[i], nu_features=n_neurons[i - 1]))

    def training(self, training_base, epochs):
        for i in range(epochs):
            for x in training_base:
                self.forward_propagation(x)
                self.back_propagation_adjust()

    # TODO
    def forward_propagation(self, input_data):
        new_input = input_data
        for layer in self.layers:
            output_data = layer.forward_process(new_input)
            new_input = output_data

    def back_propagation_adjust(self):
        pass
