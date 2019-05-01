import numpy as np
from operator import add


class NeuronioMP:
    def __init__(self, nweights=1):
        self.weights = np.random.rand(nweights)

    def predict(self, x):
        """
          Função de ativação do Perceptron Simples.

        :param x: uma linha com features de x
        :return: 1 se o somatório de ativação for positiva, caso contrário retorna 0;
        """
        activation = np.dot(self.weights.T, x).sum()
        return 1 if activation >= 0 else 0

    def training(self, training_base, epochs, learning_rate):
        """
            Realiza o treinamento do perceptron, atualizando os pesos
            com base na regra de aprendizagem.

        :param training_base: base de treino
        :param epochs: número de épocas
        :param learning_rate: taxa de aprendizagem para treinamento
        :return: época em que o erro chegou a 0 e atualiza os pesos do perceptron
        """
        epoch = 0
        for i in range(epochs):
            epoch = i
            total_error = 0
            np.random.shuffle(training_base)
            x_training = np.delete(training_base, -1, axis=1)
            y_training = training_base[:, -1]
            for j in range(x_training.shape[0]):
                prediction = self.predict(x_training[j, :])
                error = y_training[j] - prediction
                delta_w = np.array(learning_rate * error * x_training[j, :])
                self.weights = np.array(list(map(add, self.weights, delta_w)))
                total_error += abs(error)
            if total_error == 0:
                break
        return epoch

    def hit_rate(self, test_base):
        """
            Realiza o cálculo de acerto dos padrões com base nos pesos treinados.
        :param test_base: base de teste
        :return: retorna a taxa de acerto e o vetor de predicoes para o modelo treinado
        """
        x_test = np.delete(test_base, -1, 1)
        y_test = test_base[:, -1]

        predicts = []

        for i in range(x_test.shape[0]):
            prediction = self.predict(x_test[i, :])
            predicts.append(prediction)

        diff_list = list(filter(lambda x: x == 0, y_test - predicts))
        hit_rate = round((len(diff_list)/len(y_test)) * 100, 2)
        print('Hit Rate:{}%'.format(hit_rate))

        return hit_rate, predicts

