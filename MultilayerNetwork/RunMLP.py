from MultilayerNetwork.GenericMultilayerNetwork import MLP
from Utils import datasets, dataset_utils, model_utils, log_utils, plot_utils
import time
import yaml
from sklearn.model_selection import train_test_split
from copy import deepcopy
import numpy as np


def main():
    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)

    dataset, n_features = datasets.load_multiclass_base(settings['chosen_base'])
    n_labels = len(list(set(dataset[:, -1])))
    n_output_neurons = 1 if n_labels == 2 else n_labels

    epochs = settings['epochs']
    initial_learning_rate = settings['initial_learning_rate']
    final_learning_rate = settings['final_learning_rate']
    realizations = settings['realizations']

    total_execution_init = time.time()
    n_neurons = model_utils.cross_validation(5, [5, 7, 9, 11, 13], deepcopy(dataset), n_features, n_labels, settings['chosen_base'])
    execution_log = open('execution_logs/{}_{}_neurons.txt'.format(settings['chosen_base'], n_neurons), 'w')
    scenarios = []
    hit_rates = []
    execution_log.write('Base escolhida: {}\n'.format(settings['chosen_base']))
    print('Número de neurônios escohidos: {}.'.format(n_neurons))
    for i in range(realizations):
        training_base, test_base = train_test_split(dataset, test_size=settings['test_size'], stratify=dataset[:, -1])
        if n_labels > 2 and settings['chosen_base'] != '':
            training_base = dataset_utils.binarize_labels(training_base)
            test_base = dataset_utils.binarize_labels(test_base)
        start = time.time()
        mlp_test = MLP(n_layers=2, n_neurons=[n_neurons, n_output_neurons], n_features=n_features)
        mlp_test.training(training_base, epochs, initial_learning_rate=initial_learning_rate,
                          final_learning_rate=final_learning_rate)
        end = time.time()
        total_time = (end - start)
        predicted_labels, activations = mlp_test.predict(test_base)
        hit_rate = model_utils.hit_rate(predicted_labels, test_base[:, n_features+1:], activations)
        scenarios.append({'hit_rate': hit_rate,
                          'ml_net': deepcopy(mlp_test),
                          'training_base': training_base,
                          'test_base': test_base,
                          'predict': predicted_labels})
        hit_rates.append(hit_rate)
        print('Realização: {} | Tempo de treinamento:{} | Taxa de acerto: {}%'.format(i + 1, round(total_time, 2), hit_rate))
        print('----------------------------------------------------------------------')
        execution_log.write('Realização: {} | Tempo de treinamento:{} | Taxa de acerto: {}%\n'.format(i + 1, round(total_time, 2), hit_rate))
        execution_log.write('----------------------------------------------------------------------\n')

    execution_log.write('Acurácia: {}%\nDesvio Padrão:{}\n'.format(round(float(np.mean(hit_rates)), 2),
                                                                   round(float(np.std(hit_rates)), 2)))
    best_realization = log_utils.choose_realization(scenarios, settings['criterion_choiced'])
    total_execution_final = time.time()
    plot_utils.plot_conf_matrix(predict=best_realization['predict'],
                                desired_label= best_realization['test_base'][:, n_features+1:],
                                chosen_base=settings['chosen_base'],
                                n_labels=n_labels,
                                model='MLP')

    print('Tempo de execução total:{} segundos'.format(round(total_execution_final-total_execution_init)))
    execution_log.write('Tempo de execução total:{} segundos'.format(round(total_execution_final-total_execution_init)))
    execution_log.close()


if __name__ == '__main__':
    main()
