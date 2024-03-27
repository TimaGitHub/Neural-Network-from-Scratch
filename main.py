import pandas as pd
import numpy as np
from dataloader import DataBatcher
from neural_network import NeuralNetwork


if __name__ == '__main__':

    data = pd.read_csv('train.csv')

    data = np.array(data)
    data = data.astype(float)
    np.random.shuffle(data)
    test_data = data[0:1000]
    train_data = data[1000: 42000]
    test_data[:, 1:] = test_data[:, 1:] / 255
    train_data[:, 1:] = train_data[:, 1:] / 255

    test_batches = DataBatcher(test_data, 64, True)
    train_batches = DataBatcher(train_data, 64, True)

    test = NeuralNetwork(784 , [50, 200, 20] , 10,  'classification', batches = True)

    test.prepare(gradient_method = 'gd', activation_func = 'leaky_relu', seed = 42, alpha = 0.1, loss_function = 'cross_entropy_loss', optimizer = 'accelerated_momentum', momentum = 0.4)

    test.cosmetic(progress_bar = False, loss_display = True, loss_graphic = False,  iterations = 10)

    test.train(train_batches, test_batches, 3)

    test.prepare(gradient_method='gd', activation_func='leaky_relu', seed=42, alpha=0.01,
                 loss_function='cross_entropy_loss', optimizer='accelerated_momentum', momentum=0.4)

    test.cosmetic(progress_bar=False, loss_display=True, loss_graphic=False, iterations=10)




    test.train(train_batches, test_batches, 5)