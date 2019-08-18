from MultilayerNetwork.GenericMultilayerNetwork import MLP
from Utils import datasets, model_utils, log_utils, plot_utils
import time
import yaml
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)

    dataset, n_features = datasets.load_multiclass_base(settings['chosen_base'])

    epochs = settings['epochs']
    initial_learning_rate = settings['initial_learning_rate']
    final_learning_rate = settings['final_learning_rate']
    realizations = settings['realizations']

    total_execution_init = time.time()
    #n_neurons = model_utils.cross_validation(5, [5, 7, 9, 11, 13], deepcopy(dataset), n_features, n_labels, settings['chosen_base'])
    n_neurons = 2
    execution_log = open('execution_logs/{}_{}_neurons.txt'.format(settings['chosen_base'], n_neurons), 'w')
    scenarios = []
    rmses = []
    execution_log.write('Base escolhida: {}\n'.format(settings['chosen_base']))
    print('Número de neurônios escohidos: {}.'.format(n_neurons))
    for i in range(realizations):
        training_base, test_base = train_test_split(dataset, test_size=settings['test_size'])
        start = time.time()
        mlp_test = MLP(n_layers=2, n_neurons=[n_neurons, 1], n_features=n_features,
                       output_layer_func='Linear')
        mlp_test.training(training_base, epochs, initial_learning_rate=initial_learning_rate,
                          final_learning_rate=final_learning_rate)
        end = time.time()
        total_time = (end - start)
        predicted_labels, activations = mlp_test.predict(test_base)
        rmse = model_utils.rmse(predicted_labels, test_base[:, n_features+1:])
        scenarios.append({'rmse': rmse,
                          'ol_net': deepcopy(mlp_test),
                          'training_base': training_base,
                          'test_base': test_base,
                          'predicted_labels': predicted_labels})
        rmses.append(rmse)
        print('Realização: {} | Tempo de treinamento:{} | RMSE: {}'.format(i + 1, round(total_time, 2), rmse))
        print('----------------------------------------------------------------------')
        execution_log.write('Realização: {} | Tempo de treinamento:{} | RMSE: {}\n'.format(i + 1, round(total_time, 2), rmse))
        execution_log.write('----------------------------------------------------------------------\n')

    execution_log.write('RMSE médio: {}%\nDesvio Padrão:{}\n'.format(round(float(np.mean(rmses)), 2),
                                                                     float(np.std(rmses))))

    best_realization = log_utils.choose_best_realization_regression(scenarios, settings['criterion_choiced'])
    total_execution_final = time.time()

    print('Tempo de execução total:{} segundos'.format(round(total_execution_final-total_execution_init)))
    execution_log.write('Tempo de execução total:{} segundos'.format(round(total_execution_final-total_execution_init)))
    execution_log.close()
    plot_utils.plot_regression(predict=best_realization['predicted_labels'],
                               test_base=best_realization['test_base'])


if __name__ == '__main__':
    main()