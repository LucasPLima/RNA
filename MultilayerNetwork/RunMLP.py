from MultilayerNetwork.GenericMultilayerNetwork import MLP
from Utils import datasets, dataset_utils, model_utils
import time
from sklearn.model_selection import train_test_split


def main():
    dataset, n_features = datasets.load_multiclass_base('Iris')
    n_labels = len(list(set(dataset[:, -1])))
    epochs = 200
    learning_rate = 0.02
    training_base, test_base = train_test_split(dataset, test_size=0.2, stratify=dataset[:, -1])
    training_base = dataset_utils.binarize_labels(training_base)
    test_base = dataset_utils.binarize_labels(test_base)
    print('Mlp iniciado')
    start = time.time()
    mlp_test = MLP(n_layers=2, n_neurons=[9, n_labels], n_features=n_features)
    mlp_test.training(training_base, epochs, learning_rate)
    end = time.time()
    total_time = (end - start)
    predicted_labels, activations = mlp_test.predict(test_base)
    model_utils.hit_rate(predicted_labels, test_base[:, n_features+1:], activations)
    print('Mlp treinado, tempo:{}'.format(total_time))


if __name__ == '__main__':
    main()
