import time
import yaml
import Util as ut
import OneLayerNetwork.GenericOneLayerNetwork as olp
from sklearn.model_selection import train_test_split


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)
    neuron_type = settings['neuron_type']

    dataset, n_features = ut.load_multiclass_base(settings['chosen_base'], neuron_type)
    epochs = settings['epochs']
    learning_rate = settings['learning_rate']
    realizations = settings['realizations']
    scenarios = []

    total_time = 0
    for i in range(realizations):
        start = time.time()
        training_base, test_base = train_test_split(dataset, test_size=settings['test_size'], stratify=dataset[:, -1])
        training_base = ut.binarize_labels(training_base, neuron_type)
        test_base = ut.binarize_labels(test_base, neuron_type)

        perceptron_net = olp.GenericOLN(n_labels=3, n_weights=n_features + 1, neuron_type=neuron_type)
        perceptron_net.training(epochs=epochs, learning_rate=learning_rate, training_base=training_base)
        hit_rate = perceptron_net.hit_rate(test_base)

        end = time.time()
        total_time += (end - start)
        scenarios.append({'hit_rate': hit_rate,
                          'training_base': training_base,
                          'test_base': test_base})
        del perceptron_net

    print('\n--------Statistics---------')
    print('Mean execution time: {}'.format(total_time / realizations))
    best_realization = ut.choose_realization(scenarios, settings['criterion_choiced'])


if __name__ == '__main__':
    main()
