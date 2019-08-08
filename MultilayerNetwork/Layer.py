from .LogisticNeuron import LogisticNeuronML


class Layer:
    def __init__(self, is_output_layer=False, nu_neurons=5, nu_features=4, neuron_type='Logistic'):
        self.is_output_layer = is_output_layer
        self.layer_error = []
        self.neurons = [LogisticNeuronML(n_weights=nu_features + 1)
                        for i in range(nu_neurons)]
        self.layer_output = []

    def forward_process(self, data_input):
        output = [-1]
        for neuron in self.neurons:
            neuron.process_inputs(data_input)
            output.append(neuron.output)
        self.layer_output = output
        return output

    def calc_layer_error(self, desired_output=None, forward_layer=None):
        self.layer_error = []
        if desired_output is not None:
            for i in range(len(self.neurons)):
                self.layer_error.append(desired_output[i] - self.layer_output[i+1])
        else:
            for i in range(len(self.neurons)):
                error = 0
                for n in range(len(forward_layer.neurons)):
                    error += forward_layer.neurons[n].weights[i+1] * forward_layer.layer_error[n] \
                             * forward_layer.neurons[n].d_y
                self.layer_error.append(error)

    def adjust_neurons(self, learning_rate):
        for i in range(len(self.neurons)):
            self.neurons[i].adjust_weights(error=self.layer_error[i],
                                           learning_rate=learning_rate)
