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

    test.prepare(gradient_method = 'gd', activation_func = 'leaky_relu', seed = None, alpha = 0.01, loss_function = 'cross_entropy_loss', val_metric = 'accuracy',  optimizer = 'accelerated_momentum', momentum = 0.9)

    test.cosmetic(progress_bar = False, loss_display = True, loss_graphic = False,  iterations = 100)

    test.train(train_batches, test_batches, 3)

    test.prepare(gradient_method='gd', activation_func='leaky_relu', seed = None, alpha=0.001,
                 loss_function='cross_entropy_loss', val_metric = 'accuracy', optimizer='accelerated_momentum', momentum=0.9)

    test.cosmetic(progress_bar=False, loss_display=True, loss_graphic=True, iterations=100)

    test.train(train_batches, test_batches, 5)

    test.save(path = 'model_params.npy')
