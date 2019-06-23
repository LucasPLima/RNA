import OneLayerNetwork.GenericOneLayerNetwork as olp
import plot_utils as plt_ut
import dataset_utils
import log_utils
import datasets
import time
import yaml
from copy import deepcopy


from sklearn.model_selection import train_test_split


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)
    neuron_type = settings['neuron_type']

    dataset, n_features = datasets.load_multiclass_base(settings['chosen_base'])
    n_labels = len(list(set(dataset[:, -1])))
    epochs = settings['epochs']
    learning_rate = settings['learning_rate']
    realizations = settings['realizations']
    scenarios = []

    total_time = 0
    for i in range(realizations):
        start = time.time()
        training_base, test_base = train_test_split(dataset, test_size=settings['test_size'], stratify=dataset[:, -1])
        training_base = dataset_utils.binarize_labels(training_base, neuron_type)
        test_base = dataset_utils.binarize_labels(test_base, neuron_type)

        one_layer_net = olp.GenericOLN(n_labels=n_labels, n_weights=n_features + 1, neuron_type=neuron_type)
        one_layer_net.training(epochs=epochs, learning_rate=learning_rate, training_base=training_base)
        hit_rate, predict = one_layer_net.hit_rate(test_base)

        end = time.time()
        total_time += (end - start)
        print('Realization {}, Hit rate:{}%.'.format(i+1, round(hit_rate, 1)))
        scenarios.append({'hit_rate': hit_rate,
                          'ol_net': deepcopy(one_layer_net),
                          'training_base': training_base,
                          'test_base': test_base})
        del one_layer_net

    print('\n--------Statistics---------')
    print('Mean execution time: {}'.format(total_time / realizations))
    best_realization = log_utils.choose_realization(scenarios, settings['criterion_choiced'])
    plt_ut.plot_epochs_error(realization=best_realization, chosen_base=settings['chosen_base'], neuron_type=neuron_type)
    plt_ut.plot_decision_region_mult(realization=best_realization, n_classes=n_labels, choosen_base=settings['chosen_base'], neuron_type=neuron_type)


if __name__ == '__main__':
    main()
