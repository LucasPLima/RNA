import matplotlib.pyplot as plt
import numpy as np
from Utils import dataset_utils
import itertools
import yaml
from sklearn import preprocessing, datasets


def artificial_data_p():
    n_features = 2
    dataset = np.array([[np.random.uniform(0, 0.5), y, 0] for y in np.random.uniform(0, 0.5, 10)])
    dataset = np.append(dataset, [[np.random.uniform(0, 0.5), y, 0] for y in np.random.uniform(7, 7.5, 10)], axis=0)
    dataset = np.append(dataset, [[np.random.uniform(3, 3.5), y, 0] for y in np.random.uniform(0, 0.5, 10)], axis=0)
    plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
    dataset = np.append(dataset, [[np.random.uniform(3, 3.5), y, 1] for y in np.random.uniform(7, 7.5, 10)], axis=0)
    plt.plot(dataset[30:, 0], dataset[30:, 1], 'bo')

    plt.axis([-1, 4, -1, 8])
    plt.show()

    new_dataset = dataset_utils.prepare_data(dataset)

    return new_dataset, n_features


def load_multiclass_iris():
    iris = datasets.load_iris()
    stream = open('configurations/irisConfigurations.yml', 'r', encoding='utf-8').read()
    iris_cfg = yaml.load(stream=stream, Loader=yaml.FullLoader)
    n_features = len(iris_cfg['features'])

    attributes = np.array(iris['data'])

    if n_features < 4:
        chosen_features = list(set(iris_cfg['features']))
        features_indexes = sorted([iris['feature_names'].index(x) for x in chosen_features])
        new_attributes = [attributes[:, i] for i in features_indexes]
        attributes = np.array(new_attributes).T

    labels = np.array(iris['target'])

    iris_dataset = np.append(attributes, labels[:, None], axis=1)
    new_dataset = dataset_utils.prepare_data(iris_dataset)

    return new_dataset, n_features


def load_iris_dataset():
    """
       Carrega o dataset, especificando qual classe
       se deseja que seja representada pelo número 1.
       As demais se tornam 0.

       A label passada pode ser:
         0 - setosa;
         1 - versicolor;
         2 - virginica;
    """
    try:
        iris = datasets.load_iris()
        stream = open('configurations/irisConfigurations.yml', 'r', encoding='utf-8').read()
        iris_cfg = yaml.load(stream=stream, Loader=yaml.FullLoader)
        n_features = len(iris_cfg['features'])

        iris_classes = list(iris['target_names'])
        label = iris_classes.index(iris_cfg['flower_to_classify'])

        attributes = np.array(iris['data'])

        if n_features < 4:
            chosen_features = list(set(iris_cfg['features']))
            features_indexes = sorted([iris['feature_names'].index(x) for x in chosen_features])
            new_attributes = [attributes[:, i] for i in features_indexes]
            attributes = np.array(new_attributes).T

        labels = np.array(iris['target'])
        labels = np.reshape(labels, [labels.shape[0], 1])

        iris_dataset = np.append(attributes, labels, axis=1)
        new_dataset = dataset_utils.transform_dataset(iris_dataset, label)
        new_dataset = dataset_utils.prepare_data(new_dataset)

        return new_dataset, n_features

    except IndexError:
        print('Erro ao carregar Iris.')
        print('Verifique se o arquivo irisConfiguration.yml está correto.')
        exit()


def load_multiclass_artificial():
    n_features = 2
    dataset = np.array([[np.random.uniform(1, 3), y, 0] for y in np.random.uniform(4, 6, 50)])
    plt.plot(dataset[:, 0], dataset[:, 1], 'ro')
    dataset = np.append(dataset, [[np.random.uniform(4, 6), y, 1] for y in np.random.uniform(1, 3, 50)], axis=0)
    plt.plot(dataset[50:-1, 0], dataset[50:-1, 1], 'y*')
    dataset = np.append(dataset, [[np.random.uniform(7, 9), y, 2] for y in np.random.uniform(4, 6, 50)], axis=0)
    plt.plot(dataset[100:-1, 0], dataset[100:-1, 1], 'b^')

    plt.axis([0, 10, 0, 8])
    plt.savefig('plots/artificial_data_plot.png')
    plt.show()

    new_dataset = dataset_utils.prepare_data(dataset)

    return new_dataset, n_features


def load_multiclass_base(chosen_base):
    if chosen_base == 'Iris':
        return load_multiclass_iris()
    elif chosen_base == 'Artificial':
        return load_multiclass_artificial()
    else:
        print('Base escolhida não é válida.')
        print('Verifique o arquivo "runConfigurations.yml" e veja se as configurações estão corretas.')
        exit()


def load_base(chosen_base, neuron_type='L'):
    base = np.array(1)
    if chosen_base == 'Iris':
        print('Carregando configurações para Iris...')
        base = load_iris_dataset()
    elif chosen_base == 'Artificial':
        print('Carregando configurações para Artificial..')
        base = artificial_data_p()
    else:
        print('Base escolhida não é válida.')
        print('Verifique o arquivo "runConfigurations.yml" e veja se as configurações estão corretas.')
        exit()

    if neuron_type == 'H':
        for i in range(base[0].shape[0]):
            base[0][i, -1] = 1 if base[0][i, -1] == 1 else -1

    return base


def create_linear_model(n_variables, l_coeficients, n_samples, normalize):
    if len(l_coeficients) == (n_variables + 1):
        if n_variables == 1:
            x_samples = -np.ones(n_samples)
            for i in range(n_variables):
                new_row = np.linspace(-10, 10, num=n_samples)
                x_samples = np.vstack([x_samples, new_row])

            x_samples = np.vstack([x_samples, np.ones(n_samples)])
            l_coeficients.insert(0, 0)
            f_x = np.dot(l_coeficients, x_samples)

            y_samples = np.array(list(map(np.add, f_x, np.random.random(n_samples))))

            linear_model = np.vstack([x_samples, y_samples])
            linear_model = linear_model.transpose()
            x_i = list(linear_model[:, 1])
            y_i = list(linear_model[:, 3])
            min_xi = min(x_i) - 0.5
            max_xi = max(x_i) + 0.5
            min_yi = min(y_i) - 0.5
            max_yi = max(y_i) + 0.5
            plt.plot(x_i, y_i, 'ro')
            plt.axis([min_xi, max_xi, min_yi, max_yi])
            plt.show()
            if normalize:
                normalized_linear_model = np.array(preprocessing.normalize(x_samples.transpose(), norm='max', axis=0))
                normalized_linear_model = np.vstack([normalized_linear_model.transpose(), y_samples]).transpose()
                normalized_linear_model[:, 0] = -np.ones(n_samples).T
                return normalized_linear_model
            else:
                return linear_model
        if n_variables == 2:
            x_samples = np.arange(0, 10, 1, dtype=float)
            dim = x_samples.size

            linear_model = np.ones(shape=(dim * dim, 2))
            perturbation = np.random.uniform(dim * dim)
            l_coeficients.insert(0, 0)

            i = 0
            for aa, bb in itertools.product(x_samples, x_samples):
                linear_model[i][0] = aa
                linear_model[i][1] = bb
                i += 1
            y = l_coeficients[1] * linear_model[:, 0] + l_coeficients[2] * linear_model[:, 1] + l_coeficients[3] + perturbation

            linear_model = np.insert(linear_model, 0, -np.ones(dim*dim), axis=1).transpose()
            linear_model = np.vstack([linear_model, np.ones(dim*dim)])
            linear_model = np.vstack([linear_model, np.array(y)]).transpose()

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            x_i = list(linear_model[:, 1])
            y_i = list(linear_model[:, 2])
            z_i = list(linear_model[:, 4])
            ax.scatter(x_i, y_i, z_i, c='r', marker='o')
            plt.show()

            return linear_model
    else:
        print('Coeficients are lower or higher than (number_of_variables + 1)')
        return None
