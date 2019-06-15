from sklearn import preprocessing, datasets
from sklearn.metrics import confusion_matrix
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import SimplePerceptron.NeuronioPerceptron as ps
import copy
import yaml
import itertools


def prepare_data(base):
    """
        Normaliza e adiciona uma coluna de -1's ao dataset que, posteriormente,
        servirá para incluir o BIAS (w0) na somatória do perceptron.

    :param base: dataset que se deseja preparar
    :return: base de dados normalizada e com uma coluna adicional
    """
    base = base.transpose()
    normalized = [preprocessing.normalize([base[i]], norm='max') for i in range(base.shape[0] - 1)]
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

def plot_decision_region(realization, configurations, iris_cfg, class_names):
    #plt.rcParams['figure.figsize'] = (10, 7)
    plot_colors = "rb"
    plot_step = 0.01

    test_base = realization['test_base']
    training_base = realization['training_base']

    x_min, x_max = test_base[:, 1].min() - 0.2, test_base[:, 1].max() + 0.2
    y_min, y_max = test_base[:, 2].min() - 0.2, test_base[:, 2].max() + 0.2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    data = np.c_[xx.ravel(), yy.ravel()]
    bias_col = -np.ones(data.shape[0])

    data = np.insert(data, 0, bias_col, axis=1)
    perceptron_g = ps.NeuronioMP(3)
    perceptron_g.weights = realization['weights']

    Z = np.array([perceptron_g.predict(x) for x in data])
    Z = Z.reshape(xx.shape)

    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    y_test = test_base[:, -1]
    y_training = training_base[:, -1]

    if configurations['chosen_base'] == 'Iris':
        attributes = iris_cfg['features']
        plt.xlabel(attributes[0])
        plt.ylabel(attributes[1])
    else:
        plt.xlabel('Attribute x')
        plt.ylabel('Attribute y')

    plt.legend(class_names)

    for i, color in zip(range(len(class_names)), plot_colors):
        idx = np.where(y_test == i)
        plt.scatter(test_base[idx, 1], test_base[idx, 2], marker='^', c=color, label=class_names[i],
                    edgecolor='black', s=20)
        idx = np.where(y_training == i)
        plt.scatter(training_base[idx, 1], training_base[idx, 2], marker='s', c=color,
                    edgecolor='black', s=20)

    plt.show()


def choose_realization(realizations, criterion_choiced):
    """
    Escolhe a melhor realização comparando com a acurácia
    ou selecionando a taxa mais alta, com base na configuração do cenário

    :param realizations: array contendo as seguintes informações sobre cada realização:
        (taxa de acerto, array com predições, base de treino, base de teste)
    :param criterion_choiced: opção de escolha da melhor realização
                            1 - para a taxa de acerto mais alta
                            2 - para a taxa de acerto mais próxima da acurácia
    :return: a melhor realização com base na taxa de acerto média;
    """
    accuracy = 0
    hit_rates = []
    for realization in realizations:
        h = realization['hit_rate']
        accuracy += h
        hit_rates.append(h)

    accuracy = accuracy / len(realizations)
    standard_deviation = float(np.std(hit_rates))
    print('Accuracy: {}%'.format(round(accuracy, 2)))
    print('Standard Deviation: {}'.format(round(standard_deviation, 2)))

    if criterion_choiced == 1:
        best_result = max(hit_rates)
        n = hit_rates.index(best_result)
        print('Melhor realização: {} (Taxa de acerto: {}).'.format(n + 1, hit_rates[n]))
    else:
        d_means = [abs(accuracy - h) for h in hit_rates]
        nearest_accuracy = min(d_means)
        n = d_means.index(nearest_accuracy)
        print('Realização mais próxima da acurácia: {}(Taxa de acerto: {}).'.format(n + 1, hit_rates[n]))

    return realizations[n]


def plot_results(realization, configurations):
    '''
        Cria visualizações para a base escolhida. As duas visualizações criadas
        são a Matriz de confusão e as regiões de decisão
    :param realization: Dicionário contendo informações sobre a melhor realização escolhida
    :param configurations: Configurações do cenário de testes
    :return:
    '''
    plt.rcParams['figure.figsize'] = (11, 7)

    stream = open('configurations/irisConfigurations.yml', 'r', encoding='utf-8').read()
    iris_cfg = yaml.load(stream=stream, Loader=yaml.FullLoader)

    y_test = realization['test_base'][:, -1]
    y_pred = np.array(realization['predict'])

    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    if configurations['chosen_base'] == 'Iris':
        class_names = ['Iris-{}'.format(iris_cfg['flower_to_classify']), 'Others']
    else:
        class_names = ['Class 0', 'Class 1']

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names)
    plt.show()

    if len(realization['weights']) == 3:
        plot_decision_region(realization, configurations, iris_cfg, class_names)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de Confusão',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de confusão normalizada")
    else:
        print('Matriz de confusão.')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


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


def plot_adaline_results(realization, normalize):
    plt.figure()
    x = [i for i in range(len(realization['cost']))]
    plt.plot(x, realization['cost'])
    plt.show()

    fig = plt.figure()

    if normalize:
        points = np.linspace(-1, 1, 80)
    else:
        points = np.linspace(-10, 10, 80)

    weights = realization['weights']
    if len(weights) == 3:
        c_points = []
        for p in points:
            data = []
            data = [-1, p, 1]
            c_points.append(data)

        c_points = np.array(c_points)
        predict = np.dot(weights, c_points.T)

        t_base = realization['training_base']
        plt.scatter(points, predict, c='b')
        plt.scatter(t_base[:, 1], t_base[:, -1], c='r')
    else:
        t_base = np.delete(realization['training_base'], -1, axis=1)
        t_base = np.vstack([t_base, np.delete(realization['test_base'],-1, axis=1)])
        predict = np.dot(t_base, weights)
        dim= int(sqrt(t_base.shape[0]))

        plt3d = fig.gca(projection='3d')
        plt3d.cla()

        plt3d.scatter(t_base[:, 1], t_base[:, 2], predict, color='red', alpha=1.0)
        xx = np.reshape(t_base[:, 1], (dim, dim))
        yy = np.reshape(t_base[:, 2], (dim, dim))
        zz = np.reshape(predict, (dim, dim))

        plt3d.plot_surface(xx, yy, zz, rstride=10, cstride=10, antialiased=True,
                           color='blue')
        plt3d.set_xlabel('x1')
        plt3d.set_ylabel('x2')
        plt3d.set_zlabel('y')

    plt.show()

    print('Weights: {}'.format(realization['weights']))