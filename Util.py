from sklearn import preprocessing, datasets
import numpy as np
import matplotlib.pyplot as plt
import SimplePerceptron.NeuronioPerceptron as ps
import copy
import yaml


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
    new_labels = list(map(lambda x: 4 if x != label else x, new_dataset[:, -1]))
    new_labels = list(map(lambda x: 1 if x == label else 0, new_labels))
    new_labels = np.reshape(new_labels, [len(new_labels), 1])

    new_dataset = np.delete(new_dataset, -1, axis=1)
    new_dataset = np.append(new_dataset, new_labels, axis=1)

    return new_dataset


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
        new_dataset = transform_dataset(iris_dataset, label)
        new_dataset = prepare_data(new_dataset)

        return new_dataset, n_features

    except IndexError:
        print('Erro ao carregar Iris.')
        print('Verifique se o arquivo irisConfiguration.yml está correto.')
        exit()


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

    new_dataset = prepare_data(dataset)

    return new_dataset, n_features


def load_base(chosen_base):
    if chosen_base == 'Iris':
        print('Carregando configurações para Iris...')
        return load_iris_dataset()
    elif chosen_base == 'Artificial':
        print('Carregando configurações para Artificial..')
        return artificial_data_p()
    else:
        print('Base escolhida não é válida.')
        print('Verifique o arquivo "runConfigurations.yml" e veja se as configurações estão corretas.')
        exit()


def plot_decision_region(realization, configurations, iris_cfg, class_names):
    plt.rcParams['figure.figsize'] = (11, 7)
    plot_colors = "rb"
    plot_step = 0.02

    base = realization['test_base']

    x_min, x_max = base[:, 1].min() - 1, base[:, 1].max() + 1
    y_min, y_max = base[:, 2].min() - 1, base[:, 2].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    # plt.tight_layout()
    data = np.c_[xx.ravel(), yy.ravel()]
    bias_col = -np.ones(data.shape[0])

    data = np.insert(data, 0, bias_col, axis=1)
    perceptron_g = ps.NeuronioMP(3)
    perceptron_g.weights = realization['weights']

    Z = np.array([perceptron_g.predict(x) for x in data])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)
    y = base[:, -1]

    if configurations['chosen_base'] == 'Iris':
        attributes = iris_cfg['features']
        plt.xlabel(attributes[0])
        plt.ylabel(attributes[1])
    else:
        plt.xlabel('Attribute x')
        plt.ylabel('Attribute y')

    for i, color in zip(range(len(class_names)), plot_colors):
        idx = np.where(y == i)
        plt.scatter(base[idx, 1], base[idx, 2], c=color, label=class_names[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    plt.show()