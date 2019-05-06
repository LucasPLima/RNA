from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import SimplePerceptron.NeuronioPerceptron as ps
import Util as ut
import numpy as np
import time
import yaml
import itertools


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)

    base, n_features = ut.load_base(configurations['chosen_base'])

    learning_rate = configurations['learning_rate']
    iterations = configurations['realizations']
    epochs = configurations['epochs']
    realizations = []

    m_epochs = 0
    total_time = 0

    for i in range(iterations):
        start = time.time()
        training_base, test_base = train_test_split(base, test_size=configurations['test_size'])

        simplePerceptron = ps.NeuronioMP(nweights=n_features + 1)
        m_epochs += simplePerceptron.training(training_base, epochs=epochs, learning_rate=learning_rate)
        hit_rate, predict = simplePerceptron.hit_rate(test_base)

        realizations.append({'weights': simplePerceptron.weights,
                             'hit_rate': hit_rate,
                             'predict': predict,
                             'training_base': training_base,
                             'test_base': test_base})

        end = time.time()
        total_time += (end - start)
        del simplePerceptron

    print('-----------------Statistics--------------------')
    print('Mean execution time: {}'.format(total_time/iterations))
    print('Mean epochs: {}'.format(round(m_epochs/iterations)))
    best_realization = choose_realization(realizations)
    plot_results(best_realization, configurations)


def choose_realization(realizations):
    """
    Escolhe a melhor realização comparando

    :param realizations: array contendo as seguintes informações sobre cada realização:
        (taxa de acerto, array com predições, base de treino, base de teste)
    :return: a melhor realização com base na taxa de acerto média;
    """
    accuracy = 0
    standard_deviation = []
    for realization in realizations:
        hit_rate = realization['hit_rate']
        accuracy += hit_rate
        standard_deviation.append(hit_rate)

    accuracy = accuracy / len(realizations)
    print('Accuracy: {}'.format(accuracy))
    print('Standard Deviation: {}'.format(np.std(standard_deviation)))

    n = 0
    for n in range(len(realizations)):
        if realizations[n]['hit_rate'] >= accuracy:
            break

    print('Realização mais próxima da acurácia: {}.'.format(n+1))
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
        ut.plot_decision_region(realization, configurations, iris_cfg, class_names)


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


if __name__ == '__main__':
    main()
