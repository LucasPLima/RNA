import copy
import numpy as np

from sklearn import preprocessing


def prepare_data(base, n_labels=1):
    """
        Normaliza e adiciona uma coluna de -1's ao dataset que, posteriormente,
        servirá para incluir o BIAS (w0) na somatória do perceptron.

    :param base: dataset que se deseja preparar
    :param n_labels: quantidade de colunas que representam as classes
    :return: base de dados normalizada e com uma coluna adicional
    """
    base = base.transpose()
    normalized = [preprocessing.normalize([base[i]], norm='max') for i in range(base.shape[0] - n_labels)]
    normalized = np.array(normalized)

    new_base = np.reshape(normalized, (normalized.shape[0], normalized.shape[2]))
    initial_column = -np.ones((new_base.shape[1]))
    new_base = np.vstack([initial_column, new_base])
    new_base = np.vstack([new_base, base[(-n_labels):, :]])

    return new_base.transpose()


def convert_labels(labels, n_classes):
    label_binarizer = preprocessing.LabelBinarizer()
    classes = list(range(n_classes))
    label_binarizer.fit(classes)
    new_labels = np.array(label_binarizer.inverse_transform(labels))

    return new_labels


def binarize_labels(dataset, neuron_type='L'):
    labels = dataset[:, -1]
    classes = list(set(labels))

    multi_bin = preprocessing.LabelBinarizer()
    multi_bin.fit(classes)
    binarized_labels = np.array(multi_bin.transform(labels))

    if neuron_type == 'H':
        binarized_labels = np.where(binarized_labels == 0, -1, binarized_labels)

    new_dataset = np.delete(dataset, -1, axis=1)
    new_dataset = np.append(new_dataset, binarized_labels, axis=1)

    return new_dataset


def transform_dataset(dataset, label):
    """
    Muda as label da classe escolhida para 1 e considera as outras como 0.

    :param dataset: base de dados
    :param label: label que se deseja transformar em 1
    :return: um novo dataset com as labels alteradas
    """
    new_dataset = copy.deepcopy(dataset)
    new_labels = list(map(lambda x: 4 if x != label else x, new_dataset[:, -1]))
    new_labels = list(map(lambda x: 1 if x == label else 0, new_labels))
    new_labels = np.reshape(new_labels, [len(new_labels), 1])

    new_dataset = np.delete(new_dataset, -1, axis=1)
    new_dataset = np.append(new_dataset, new_labels, axis=1)

    return new_dataset
