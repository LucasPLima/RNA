import numpy as np


def validate_predict(predict, activations):
    max_activation = max(activations)
    if sum(predict) == 0:
        predict[activations.index(max_activation)] = 1
    elif sum(predict) > 0:
        predict = [0 for i in range(len(predict))]
        predict[activations.index(max_activation)] = 1


def hit_rate(predicted_labels, desired_labels, activations=None):
    hit = []
    for i in range(len(predicted_labels)):
        validate_predict(predicted_labels[i], activations[i])
        error = sum(desired_labels[i] - predicted_labels[i])
        if error == 0:
            hit.append(1)
        #hit = list(filter(lambda x: sum(x) == 0, ))

    rate = (len(hit) / desired_labels.shape[0]) * 100
    print('Hit Rate: {}%'.format(rate))
