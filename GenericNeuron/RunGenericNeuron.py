import yaml
import time
import Util as ut
import GenericNeuron.LogisticNeuron as logisticNeuron
import GenericNeuron.HiperbolicNeuron as hiperbolicNeuron
from sklearn.model_selection import train_test_split


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)

    base, n_features = ut.load_base(configurations['chosen_base'], configurations['neuron_type'])

    realizations = configurations['realizations']
    learning_rate = configurations['learning_rate']
    epochs = configurations['epochs']
    scenarios = []
    total_time = 0

    for i in range(realizations):
        start = time.time()
        training_base, test_base = train_test_split(base, test_size=configurations['test_size'], stratify=base[:, -1])

        if configurations['neuron_type'] == 'L':
            generic_neuron = logisticNeuron.LogisticNeuron(n_weights=n_features + 1)
        elif configurations['neuron_type'] == 'H':
            generic_neuron = hiperbolicNeuron.HiperbolicNeuron(n_weights=n_features + 1)

        generic_neuron.training(training_base=training_base, learning_rate=learning_rate, n_epochs=epochs)
        generic_neuron.hit_rate(test_base)
        end = time.time()
        total_time += (end-start)
        '''
        scenarios.append({'MSE': mse,
                             'RMSE':rmse,
                             'weights':AdalineGD.weights,
                             'cost': AdalineGD.cost,
                             'training_base':training_base,
                             'test_base': test_base})
        '''

    print('\n--------Statistics---------')
    print('Mean execution time: {}'.format(total_time/realizations))


if __name__ == '__main__':
    main()