from sklearn.model_selection import train_test_split
from Utils import plot_utils as plt_ut, datasets
import numpy as np
import Adaline.GradientDescendentAdaline as gd
import time
import yaml


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    configurations = yaml.load(stream=stream, Loader=yaml.FullLoader)

    # linear model 1: z = 2x + 3 + noise
    base = datasets.create_linear_model(n_variables=configurations['n_variables'],
                                        l_coeficients=configurations['coeficients'],
                                        n_samples=configurations['n_samples'],
                                        normalize= bool(configurations['normalize']))

    iterations = configurations['realizations']
    learning_rate = configurations['learning_rate']
    epochs = configurations['epochs']
    realizations = []
    total_time = 0

    for i in range(iterations):
        start = time.time()
        training_base, test_base = train_test_split(base, test_size=configurations['test_size'])

        AdalineGD = gd.Adaline(n_weights=len(configurations['coeficients']))
        AdalineGD.training(training_base, epochs=epochs, learning_rate=learning_rate) #random.uniform(0.001, 0.010)
        print('Iteration {}'.format(i+1))
        print(AdalineGD.weights)
        mse, rmse = AdalineGD.test(test_base)
        print('------------------------')
        end = time.time()
        total_time += (end-start)
        realizations.append({'MSE': mse,
                             'RMSE':rmse,
                             'weights':AdalineGD.weights,
                             'cost': AdalineGD.cost,
                             'training_base':training_base,
                             'test_base': test_base})

    print('\n--------Statistics---------')
    print('Mean execution time: {}'.format(total_time/iterations))
    realization = choose_best_realization(realizations)
    plt_ut.plot_adaline_results(realization, bool(configurations['normalize']))


def choose_best_realization(realizations):
    m_mse = []
    m_rmse = []
    for i in range(len(realizations)):
        m_mse.append(realizations[i]['MSE'])
        m_rmse.append(realizations[i]['RMSE'])
    min_mse = min(m_mse)

    n = m_mse.index(min_mse)

    stand_deviation_mse = np.std(m_mse)
    stand_deviation_rmse = np.std(m_rmse)

    print('\nBest realization: {}'.format(n+1))
    print('Mean of MSE:{}'.format(np.mean(m_mse)))
    print('MSE of best realization: {}'.format(min_mse))
    print('RMSE of best realization: {} \n'.format(realizations[n]['RMSE']))

    print('Standard Deviation MSE: {}'.format(stand_deviation_mse))
    print('Standard Deviation RMSE: {}'.format(stand_deviation_rmse))

    return realizations[n]


if __name__ == '__main__':
    main()
