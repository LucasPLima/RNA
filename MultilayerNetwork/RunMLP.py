from MultilayerNetwork.GenericMultilayerNetwork import MLP
from Utils import datasets, dataset_utils, model_utils
import time
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    dataset, n_features = datasets.load_multiclass_base('Iris')
    n_labels = len(list(set(dataset[:, -1])))
    epochs = 200
    initial_learning_rate = 0.5
    final_learning_rate = 0.1
    realizations = 20
    hit_rates = []
    for i in range(realizations):
        training_base, test_base = train_test_split(dataset, test_size=0.2, stratify=dataset[:, -1])
        training_base = dataset_utils.binarize_labels(training_base)
        test_base = dataset_utils.binarize_labels(test_base)
        start = time.time()
        mlp_test = MLP(n_layers=2, n_neurons=[6, n_labels], n_features=n_features)
        mlp_test.training(training_base, epochs, initial_learning_rate=initial_learning_rate,
                          final_learning_rate=final_learning_rate)
        end = time.time()
        total_time = (end - start)
        predicted_labels, activations = mlp_test.predict(test_base)
        hit_rate = model_utils.hit_rate(predicted_labels, test_base[:, n_features+1:], activations)
        hit_rates.append(hit_rate)
        print('Realizacação: {} | Tempo de treinamento:{} | Hit rate: {}%'.format(i + 1, round(total_time, 2), hit_rate))
        print('---------------------------------------------------------')

    accuracy = float(np.mean(hit_rates))
    print('Acurácia: {}%'.format(round(accuracy, 2)))
    # TODO
    # printar região de decisão


if __name__ == '__main__':
    main()
