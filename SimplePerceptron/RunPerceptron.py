from sklearn.model_selection import train_test_split
import SimplePerceptron.NeuronioPerceptron as ps
from Utils import plot_utils as plt_ut, datasets, log_utils
import time
import yaml


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)

    base, n_features = datasets.load_base(configurations['chosen_base'])

    learning_rate = configurations['learning_rate']
    iterations = configurations['realizations']
    epochs = configurations['epochs']
    realizations = []

    m_epochs = 0
    total_time = 0

    for i in range(iterations):
        start = time.time()
        training_base, test_base = train_test_split(base, test_size=configurations['test_size'], stratify=base[:, -1])

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
    best_realization = log_utils.choose_realization(realizations, configurations['criterion_choiced'])
    plt_ut.plot_results(best_realization, configurations)


if __name__ == '__main__':
    main()
