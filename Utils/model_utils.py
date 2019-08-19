import yaml
import numpy as np
from MultilayerNetwork import GenericMultilayerNetwork as gmn
from MultilayerNetwork import ExtremeLearningMachine as elm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from Utils import dataset_utils
from math import sqrt


def validate_predict(predict, activations):
    max_activation = max(activations)
    new_predict = predict
    if len(predict) > 1:
        if sum(predict) == 0:
            new_predict[activations.index(max_activation)] = 1
        elif sum(predict) > 1:
            new_predict = [0 for i in range(len(predict))]
            new_predict[activations.index(max_activation)] = 1

    return new_predict


def hit_rate(predicted_labels, desired_labels, activations=None):
    hit = []
    for i in range(len(predicted_labels)):
        new_predict = validate_predict(predicted_labels[i], activations[i])
        error = desired_labels[i] - new_predict
        if sum(error**2) == 0:
            hit.append(1)

    rate = (len(hit) / desired_labels.shape[0]) * 100
    return round(rate, 2)


def eta_decay(actual_epoch, final_epoch, initial_learning_rate, final_learning_rate):
    new_eta = initial_learning_rate * ((final_learning_rate / initial_learning_rate)**(actual_epoch/final_epoch))
    return new_eta


def rmse(predicted_labels, desired_labels):
    mean_se = mean_squared_error(desired_labels, predicted_labels)
    rad_mse = sqrt(mean_se)
    return rad_mse


def cross_validation(splits, n_hidden_neurons, base, n_features, n_classes, chosen_base, model='MLP'):
    print('Cross Validation init:')

    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)
    log_file = open('grid_search_log/{}_{}_log.txt'.format(model, chosen_base), 'w')

    folds = KFold(n_splits=splits)
    np.random.shuffle(base)
    grid_search_base = base[0:int(round((1-settings['test_size'])*base.shape[0])), :]
    n_output_neurons = 1 if n_classes == 2 else n_classes

    if n_classes > 2:
        grid_search_base = dataset_utils.binarize_labels(grid_search_base)
    accuracy = []
    for n in range(len(n_hidden_neurons)):
        realization = 0
        hit_rates = []
        print('Number of hidden neurons:{}'.format(n_hidden_neurons[n]))
        log_file.write('Number of hidden neurons:{}\n'.format(n_hidden_neurons[n]))
        for train_index, test_index in folds.split(grid_search_base):
            training_base = grid_search_base[train_index, :]
            test_base = grid_search_base[test_index, :]
            mlp = gmn.MLP(2, [n_hidden_neurons[n], n_output_neurons], n_features)
            mlp.training(training_base, epochs=settings['epochs'],initial_learning_rate=settings['initial_learning_rate'],
                         final_learning_rate=settings['final_learning_rate'])
            predicted_labels, activations = mlp.predict(test_base)
            hit = hit_rate(predicted_labels, test_base[:, n_features + 1:], activations)
            print('\tHit rate for realization {}: {}%'.format(realization + 1, hit))
            log_file.write('\tHit rate for realization {}: {}%\n'.format(realization + 1, hit))
            hit_rates.append(hit)
            realization += 1
            del mlp
        accuracy.append(np.mean(hit_rates))
        print('\tAccuracy with {} hidden neurons: {}%'.format(n_hidden_neurons[n], round(accuracy[n], 2)))
        log_file.write('\tAccuracy with {} hidden neurons: {}%\n'.format(n_hidden_neurons[n], round(accuracy[n], 2)))

    print('Cross Validation end.')
    log_file.close()
    best_realization = max(accuracy)
    return n_hidden_neurons[accuracy.index(best_realization)]


def cross_validation_elm(splits, n_hidden_neurons, base, n_features, n_classes, chosen_base, model='ELM'):
    print('Cross Validation init:')

    stream = open('configurations/runConfigurations.yml', 'r', encoding='utf-8').read()
    settings = yaml.load(stream=stream, Loader=yaml.FullLoader)
    log_file = open('grid_search_log/{}_{}_log.txt'.format(model, chosen_base), 'w')

    folds = KFold(n_splits=splits)
    np.random.shuffle(base)
    grid_search_base = base[0:int(round((1 - settings['test_size']) * base.shape[0])), :]
    n_output_neurons = 1 if n_classes == 2 else n_classes

    if n_classes > 2:
        grid_search_base = dataset_utils.binarize_labels(grid_search_base)
    accuracy = []
    for n in range(len(n_hidden_neurons)):
        realization = 0
        hit_rates = []
        print('Number of hidden neurons:{}'.format(n_hidden_neurons[n]))
        log_file.write('Number of hidden neurons:{}\n'.format(n_hidden_neurons[n]))
        for train_index, test_index in folds.split(grid_search_base):
            training_base = grid_search_base[train_index, :]
            test_base = grid_search_base[test_index, :]
            elm_test = elm.ELM(2, [n_hidden_neurons[n], n_output_neurons], n_features)
            elm_test.training(training_base)
            predicted_labels, activations = elm_test.predict(test_base)
            hit = hit_rate(predicted_labels, test_base[:, n_features + 1:], activations)
            print('\tHit rate for realization {}: {}%'.format(realization + 1, hit))
            log_file.write('\tHit rate for realization {}: {}%\n'.format(realization + 1, hit))
            hit_rates.append(hit)
            realization += 1
            del elm_test
        accuracy.append(np.mean(hit_rates))
        print('\tAccuracy with {} hidden neurons: {}%'.format(n_hidden_neurons[n], round(accuracy[n], 2)))
        log_file.write('\tAccuracy with {} hidden neurons: {}%\n'.format(n_hidden_neurons[n], round(accuracy[n], 2)))

    print('Cross Validation end.')
    log_file.close()
    best_realization = max(accuracy)
    return n_hidden_neurons[accuracy.index(best_realization)]
