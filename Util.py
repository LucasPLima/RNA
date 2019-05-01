import numpy as np
from sklearn import preprocessing, datasets
import copy


def prepare_data(base):
    """
        Normaliza e adiciona uma coluna de -1's ao dataset que, posteriormente,
        servirá para incluir o BIAS (w0) na somatória do perceptron.

    :param base: dataset que se deseja preparar
    :return: base de dados normalizada e com uma coluna adicional
    """
    base = base.transpose()
    normalized = [preprocessing.normalize([base[i]]) for i in range(base.shape[0] - 1)]
    normalized = np.array(normalized)

    new_base = np.reshape(normalized, (normalized.shape[0], normalized.shape[2]))
    initial_column = -np.ones((new_base.shape[1]))
    new_base = np.vstack([initial_column, new_base])
    new_base = np.vstack([new_base, base[-1, :]])

    return new_base.transpose()


def transform_dataset(dataset, label):
    """
    Muda as label da classe escolhida para 1 e considera as outras como 0.

    :param dataset: base de dados
    :param label: label que se deseja transformar em 1
    :return: um novo dataset com as labels alteradas
    """
    new_dataset = copy.deepcopy(dataset)
    new_labels = list(map(lambda x: 4 if x != label else x, new_dataset[:, 4]))
    new_labels = list(map(lambda x: 1 if x == label else 0, new_labels))
    new_labels = np.reshape(new_labels, [len(new_labels), 1])

    new_dataset = np.delete(new_dataset, -1, axis=1)
    new_dataset = np.append(new_dataset, new_labels, axis=1)

    return new_dataset


def load_iris_dataset(label):
    """
       Carrega o dataset, especificando qual classe
       se deseja que seja representada pelo número 1.
       As demais se tornam 0.

       A label passada pode ser:
         0 - setosa;
         1 - versicolor;
         2 - virginica;
    """
    iris = datasets.load_iris()
    iris_features = iris['feature_names']
    attributes = np.array(iris['data'])
    labels = np.array(iris['target'])
    labels = np.reshape(labels, [labels.shape[0], 1])

    iris_dataset = np.append(attributes, labels, axis=1)
    new_dataset = transform_dataset(iris_dataset, label)
    new_dataset = prepare_data(new_dataset)

    return new_dataset, len(iris_features)

