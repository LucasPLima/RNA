import SimplePerceptron.NeuronioPerceptron as ps
import matplotlib.pyplot as plt
import numpy as np
from Utils import dataset_utils
import yaml
import itertools

from sklearn.metrics import confusion_matrix
from math import sqrt


def plot_decision_region(realization, configurations, iris_cfg, class_names):
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


def plot_results(realization, configurations):
    """
        Cria visualizações para a base escolhida. As duas visualizações criadas
        são a Matriz de confusão e as regiões de decisão
    :param realization: Dicionário contendo informações sobre a melhor realização escolhida
    :param configurations: Configurações do cenário de testes
    :return:
    """
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
            data = [-1, p, 1]
            c_points.append(data)

        c_points = np.array(c_points)
        predict = np.dot(weights, c_points.T)

        t_base = realization['training_base']
        plt.scatter(points, predict, c='b')
        plt.scatter(t_base[:, 1], t_base[:, -1], c='r')
    else:
        t_base = np.delete(realization['training_base'], -1, axis=1)
        t_base = np.vstack([t_base, np.delete(realization['test_base'], -1, axis=1)])
        predict = np.dot(t_base, weights)
        dim = int(sqrt(t_base.shape[0]))

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


def plot_epochs_error(realization, chosen_base, neuron_type):
    ol_net = realization['ol_net']
    epochs = list(range(len(ol_net.epochs_error)))
    epochs = [i+1 for i in epochs]

    ax = plt.gca()

    colors = 'ryb'
    for i in range(len(ol_net.neurons)):
        plt.plot(epochs, ol_net.neurons[i].cost, label='Neuron {}'.format(i+1),
                 color=colors[i], linewidth=2.5)

    ax.set(xlabel='Epochs', ylabel='Error',
           title='Total cost per epoch.')
    ax.grid()
    ax.legend()
    plt.savefig('plots/{}_{}_epochs_error.png'.format(chosen_base, neuron_type))
    plt.show()


def plot_decision_region_mult(realization, n_classes, choosen_base, neuron_type):
    plt.rcParams['figure.figsize'] = (11, 7)
    plot_colors = "ryb"
    plot_step = 0.01

    generic_ol_net = realization['ol_net']
    base = np.append(realization['training_base'], realization['test_base'], axis=0)
    X_test = realization['test_base'][:, 0:(-n_classes)]
    X_sample = base[:, 0:(-n_classes)]
    Y_sample = dataset_utils.convert_labels(base[:, -n_classes:], n_classes)

    if X_sample.shape[1] == 3:
        x_min, x_max = X_sample[:, 1].min() - 0.5, X_sample[:, 1].max() + 0.5
        y_min, y_max = X_sample[:, 2].min() - 0.5, X_sample[:, 2].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        data = np.c_[xx.ravel(), yy.ravel()]
        bias_col = -np.ones(data.shape[0])
        data = np.insert(data, 0, bias_col, axis=1)

        Z = np.array([generic_ol_net.predict(x) for x in data])
        Z = dataset_utils.convert_labels(labels=Z, n_classes=n_classes)
        Z = Z.reshape(xx.shape)

        cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

        classes = []
        if choosen_base == 'Artificial':
            plt.xlabel('Attribute 1')
            plt.ylabel('Attribute 2')
            classes = ['Ball Class', 'Star Class', 'Triangle Class']
        elif choosen_base == 'Iris':
            stream = open('configurations/irisConfigurations.yml', 'r', encoding='utf-8').read()
            iris_cfg = yaml.load(stream=stream, Loader=yaml.FullLoader)
            plt.xlabel(iris_cfg['features'][0])
            plt.ylabel(iris_cfg['features'][1])
            classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(Y_sample == i)
            plt.scatter(X_sample[idx, 1], X_sample[idx, 2], c=color, label=classes[i],
                        edgecolor='black', s=25)
        plt.legend()

        for i in range(X_test.shape[0]):
            plt.plot(X_test[i, 1], X_test[i, 2], 'ko', fillstyle='none', markersize=8)

        plt.savefig('plots/{}_{}_decision_region.png'.format(choosen_base, neuron_type))
        plt.show()
    else:
        print("Can't plot decision region for a dataset with more than 2 features.")
