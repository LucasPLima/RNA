from MultilayerNetwork.GenericMultilayerNetwork import MLP
from Utils import datasets, dataset_utils
import time
import yaml
from copy import deepcopy
from sklearn.model_selection import train_test_split


def main():
    dataset, n_features = datasets.load_multiclass_base('Iris')
    n_labels = len(list(set(dataset[:, -1])))
    epochs = 100
    learning_rate = 0.02
    training_base, test_base = train_test_split(dataset, test_size=0.2, stratify=dataset[:, -1])
    training_base = dataset_utils.binarize_labels(training_base)
    test_base = dataset_utils.binarize_labels(test_base)
    print('Mlp iniciado')
    start = time.time()
    mlp_test = MLP(n_layers=2, n_neurons=[5, n_labels], n_features=n_features)
    print('Pesos Iniciais')
    for neuron in mlp_test.layers[-1].neurons:
        print(neuron.weights)
    mlp_test.training(training_base, epochs, learning_rate)
    print('Pesos Finais')
    for neuron in mlp_test.layers[-1].neurons:
        print(neuron.weights)
    end = time.time()
    total_time = (end - start)
    print('Mlp treinado, tempo:{}'.format(total_time))


if __name__ == '__main__':
    main()
