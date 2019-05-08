from sklearn import preprocessing, datasets
from sklearn.metrics import confusion_matrix
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
                    edgecolor='black', s=15)
        idx = np.where(y_training == i)
        plt.scatter(training_base[idx, 1], training_base[idx, 2], marker='s', c=color,
                    edgecolor='black', s=15)

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
        print('Melhor realização: {}'.format(n+1))
    else:
        d_means = [accuracy - h for h in hit_rates]
        nearest_accuracy = min(d_means)
        n = d_means.index(nearest_accuracy)
        print('Realização mais próxima da acurácia: {}.'.format(n + 1))

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

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()