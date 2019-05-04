from sklearn.model_selection import train_test_split
import time
import Util as ut
import SimplePerceptron.NeuronioPerceptron as ps
import yaml


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
    #plot_results(best_realization)


def choose_realization(realizations):
    """
    Escolhe a melhor realização comparando

    :param realizations: array contendo as seguintes informações sobre cada realização:
        (taxa de acerto, array com predições, base de treino, base de teste)
    :return: a melhor realização com base na taxa de acerto média;
    """
    accuracy = 0
    for realization in realizations:
        accuracy += realization['hit_rate']

    accuracy = accuracy / len(realizations)
    print('Accuracy: {}'.format(accuracy))

    n = 0
    for n in range(len(realizations)):
        if realizations[n]['hit_rate'] >= accuracy:
            break
    return realizations[n]


if __name__ == '__main__':
    main()
