import pandas as pd
import numpy as np

from neural_network import NeuralNetwork


if __name__ == '__main__':

    data = pd.read_csv('train.csv')

    data = np.array(data)
    m, n = data.shape

    test_data = data[0:1000]
    digit_test = test_data[:, 0]
    digit_test.shape = -1, 1
    digit_test = np.int_(np.arange(0,10) == digit_test)
    param_test = test_data[:, 1:n]
    param_test = param_test / 255

    train_data = data[1000: 42000]
    digit = train_data[:, 0]
    digit.shape = -1, 1
    digit = np.int_(np.arange(0, 10) == digit)
    param = train_data[:, 1:n]
    param = param / 255

    test = NeuralNetwork(2, 20, 'classification')
    test.prepare(gradient_method = 'sagd', activation_func = 'leaky_relu', seed = None, alpha = 0.1, loss_function = 'cross_entropy_loss')
    test.cosmetic(progress_bar = True, loss_display = True, iterations = 10)
    test.train(param, digit, 100)