from MultilayerNetwork.LogisticNeuron import LogisticNeuronML


class Layer:
    def __init__(self, is_output_layer=False, nu_neurons=5, nu_features=4, neuron_type='Logistic'):
        self.is_output_layer = is_output_layer
        self.layer_error = []
        self.neurons = [LogisticNeuronML(n_weights=nu_features + 1)
                        for i in range(nu_neurons)]
        self.layer_output = []

    def forward_process(self, data_input):
        output = []
        for neuron in self.neurons:
            neuron.process_inputs(data_input)
            output.append(neuron.output)
        self.layer_output = output
        return output

    def calc_layer_error(self):
        pass
