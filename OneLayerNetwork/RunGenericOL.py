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

    total_time = 0
    for i in range(realizations):
        start = time.time()
        training_base, test_base = train_test_split(dataset, test_size=settings['test_size'])

        perceptron_net = olp.GenericOLN(n_labels=3, n_weights=n_features + 1, neuron_type=neuron_type)
        perceptron_net.training(epochs=epochs, learning_rate=learning_rate, training_base=training_base)
        perceptron_net.hit_rate(test_base)

        end = time.time()
        total_time += (end - start)
        del perceptron_net

    print('\n--------Statistics---------')
    print('Mean execution time: {}'.format(total_time / realizations))


if __name__ == '__main__':
    main()
