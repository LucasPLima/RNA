import numpy as np


def choose_realization(realizations, criterion_choiced):
    """
    Escolhe a melhor realização comparando com a acurácia
    ou selecionando a taxa mais alta, com base na configuração do cenário

    :param realizations: array contendo as seguintes informações sobre cada realização:
        (taxa de acerto, array com predições, base de treino, base de teste)
    :param criterion_choiced: opção de escolha da melhor realização
                            1 - para a taxa de acerto mais alta
                            2 - para a taxa de acerto mais próxima da acurácia
                            3 - para a taxa de acerto mais baixa
    :return: a melhor realização com base na taxa de acerto média;
    """
    accuracy = 0
    hit_rates = []
    for realization in realizations:
        h = realization['hit_rate']
        accuracy += h
        hit_rates.append(h)

    accuracy = accuracy / len(realizations)
    standard_deviation = float(np.std(hit_rates))
    print('Accuracy: {}%'.format(round(accuracy, 2)))
    print('Standard Deviation: {}'.format(round(standard_deviation, 2)))

    if criterion_choiced == 1:
        best_result = max(hit_rates)
        n = hit_rates.index(best_result)
        print('Melhor realização: {} (Taxa de acerto: {}).'.format(n + 1, hit_rates[n]))
    elif criterion_choiced == 2:
        d_means = [abs(accuracy - h) for h in hit_rates]
        nearest_accuracy = min(d_means)
        n = d_means.index(nearest_accuracy)
        print('Realização mais próxima da acurácia: {}(Taxa de acerto: {}).'.format(n + 1, round(hit_rates[n], 2)))
    else:
        worst_result = min(hit_rates)
        n = hit_rates.index(worst_result)
        print('Pior realização: {} (Taxa de acerto: {})'.format(n+1, hit_rates[n]))

    return realizations[n]