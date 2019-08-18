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
    print('Acurácia: {}%'.format(round(accuracy, 2)))
    print('Desvio Padrão: {}'.format(round(standard_deviation, 2)))

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


def choose_best_realization_regression(realizations, criterion_choiced):
    m_rmse = []

    for i in range(len(realizations)):
        m_rmse.append(realizations[i]['rmse'])

    print('RMSE Médio: {}'.format(round(float(np.mean(m_rmse)), 4)))
    print('Desvio Padrão: {}'.format(np.std(m_rmse)))

    if criterion_choiced == 1:
        best_result = min(m_rmse)
        n = m_rmse.index(best_result)
        print('Melhor realização: {} (RMSE: {}).'.format(n + 1, m_rmse[n]))
    elif criterion_choiced == 2:
        mean_rmse = round(float(np.mean(m_rmse)), 2)
        d_means = [abs(mean_rmse - h) for h in m_rmse]
        nearest_accuracy = min(d_means)
        n = d_means.index(nearest_accuracy)
        print('Realização mais próxima da média: {}(RMSE: {}).'.format(n + 1, round(m_rmse[n], 2)))
    else:
        worst_result = max(m_rmse)
        n = m_rmse.index(worst_result)
        print('Pior realização: {} (RMSE: {})'.format(n+1, m_rmse[n]))

    print('\nRealização escolhida: {}'.format(n + 1))
    print('RMSE da realização escolhida: {} \n'.format(realizations[n]['rmse']))

    return realizations[n]
