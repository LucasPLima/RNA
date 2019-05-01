from sklearn.model_selection import train_test_split
import time
import Util as ut
import SimplePerceptron.NeuronioPerceptron as ps


def main():
    base, n_features = ut.load_iris_dataset(0)
    learning_rate = 0.03
    iterations = 20
    epochs = 300
    realizations = []

    m_epochs = 0
    total_time = 0

    for i in range(iterations):
        start = time.time()
        training_base, test_base = train_test_split(base, test_size=0.2)

        simplePerceptron = ps.NeuronioMP(nweights= n_features +1)
        m_epochs += simplePerceptron.training(training_base, epochs=epochs, learning_rate=learning_rate)
        accuracy, predict = simplePerceptron.hit_rate(test_base)
        realizations.append((accuracy, predict, training_base, simplePerceptron.weights, test_base))

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
    for i in realizations:
        accuracy += i[0]

    accuracy = accuracy / len(realizations)
    print('Mean accuracy: {}'.format(accuracy))

    n = 0
    for n in range(len(realizations)):
        if realizations[n][0] >= accuracy:
            break
    return realizations[n]


if __name__ == '__main__':
    main()
