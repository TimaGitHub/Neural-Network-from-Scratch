import functions
import metrics
import gradient_steps
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import matplotlib
matplotlib.use("TkAgg")

class NeuralNetwork():

    def __init__(self, n_inputs, neurons, n_outputs, purpose='classification', batches=False):

        self.n_inputs = n_inputs
        self.neurons = neurons
        self.purpose = purpose
        self.n_outputs = n_outputs
        self.trained = False
        self.history_scores = []
        self.history_losses = []

        # The next 4 lines of code are needed in case you want to load pre-trained model parameters
        self.weights = [0] * (len(neurons) + 2)
        self.biases = [0] * (len(neurons) + 1)
        self.last_grad_w = [0] * len(self.weights)
        self.last_grad_b = [0] * len(self.biases)

    def prepare(self, gradient_method='gd', activation_func='sigmoid', alpha=0.1, seed=None,
                loss_function='cross_entropy_loss', val_metric = 'accuracy', optimizer=False,  momentum = 0):

        np.random.seed(seed)

        self.alpha = alpha
        self.activation_func, self.derivative = functions.get_func(activation_func)
        self.loss_func, self.loss_derivative = functions.get_loss_func(loss_function)
        self.val_metric = metrics.accuracy if val_metric == 'accuracy' else metrics.auc
        self.metric_name = val_metric

        if gradient_method == 'gd':
            self.gradient_method = gradient_steps.gradient_descent

        elif gradient_method == 'sgd':
            self.gradient_method = gradient_steps.stochastic_gradient_descent

        elif gradient_method == 'sagd':
            self.gradient_method = gradient_steps.stochastic_average_gradient_descent

        else:
            raise Exception("gradient descent method is not specified or unknown")

        if optimizer == 'accelerated_momentum':
            self.optimizer = 'accelerated_momentum'
            self.momentum = momentum

        elif optimizer == False:
            self.optimizer = None
            self.momentum = 0

        else:
            raise Exception("Optimizer is not specified or unknown")

    def cosmetic(self, progress_bar=False, loss_display=False, loss_graphic = False, iterations=0):
        # printing this "For epoch number: ..., validation accuracy is: ..., loss is ..."
        self.loss_display = loss_display

        # to depict the learning process.
        self.loss_graphic = loss_graphic

        # how often you would like to get message about training process
        self.iterations = iterations

        if not progress_bar:
            def tqdm_False(x, **params__):
                return x

            self.tqdm = tqdm_False
        else:
            self.tqdm = tqdm

    def train(self, batches, test_batches, n_epochs):
        try:
            if self.purpose == 'classification':

                if not self.trained:
                    # if not trained, let's initialize new parameters of the model
                    for index, neuron in enumerate(self.neurons):

                        if index == 0:

                            self.weights = [np.random.uniform(- 0.5, 0.5, size=(self.n_inputs, neuron))]

                            self.biases = [np.random.uniform(- 0.5, 0.5, size=(1, neuron))]


                        else:

                            self.weights.append(np.random.uniform(-0.5, 0.5, size=(last, neuron)))

                            self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, neuron)))

                        last = neuron + 0

                    self.weights.append(np.random.uniform(-0.5, 0.5, size=(last, self.n_outputs)))

                    self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, self.n_outputs)))

                    # add identity matrix for correct chain rule algorithm
                    self.weights.append(np.eye(self.n_outputs, self.n_outputs))

                    self.trained = True

                for epoch in self.tqdm(range(n_epochs), position=0, leave=True):

                    for index, batch in enumerate(batches):

                        x_train, y_train = batch
                        # collect outputs from different layers for back propagation
                        self.hidden_outputs_no_activation = []
                        self.hidden_outputs_activation = []

                        self.hidden_outputs_activation.append(x_train)
                        self.hidden_outputs_no_activation.append(x_train)


                        # main forward-back-propagation process begins here

                        result = x_train @ self.weights[0] + self.biases[0]
                        self.hidden_outputs_no_activation.append(result)
                        self.hidden_outputs_activation.append(self.activation_func(result))

                        for i in range(len(self.neurons)):

                            result = self.hidden_outputs_activation[-1] @ self.weights[i + 1] + self.biases[i + 1]

                            self.hidden_outputs_no_activation.append(result)

                            self.hidden_outputs_activation.append(self.activation_func(result))

                        # remove last result to apply softmax function
                        pop_garbage = self.hidden_outputs_activation.pop()

                        output = functions.softmax(result)
                        self.hidden_outputs_activation.append(output)

                        # calculation loss for back-propagation
                        loss = self.loss_func(output, y_train)
                        self.gradient_method(self, output, y_train)

                        if self.loss_display and index % self.iterations == 0:
                            val_acc = []
                            for batch in test_batches:
                                x_train, y_train = batch
                                val_acc.append(self.val_metric(np.int_(np.arange(0, 10) == y_train), self.predict(x_train)))
                            print('For epoch number: {}, validation {} is: {}, loss is {}'.format(epoch, self.metric_name, round(
                                np.mean(val_acc), 4), round(np.mean(loss), 4)))


                            self.history_losses.append(np.mean(loss))
                            self.history_scores.append(np.mean(val_acc))
                            if self.loss_graphic:
                                pass
                                # if you have notebook uncomment this block of code and comment 'pass'
                                # fig, ax1 = plt.subplots(figsize=(9, 8))
                                #
                                # clear_output(True)
                                #
                                # ax1.set_xlabel('iters')
                                #
                                # ax1.set_ylabel('Loss', color='blue')
                                #
                                # t = np.arange(len(self.history_losses))
                                #
                                # ax1.plot(t, self.history_losses)
                                #
                                # ax2 = ax1.twinx()
                                #
                                # ax2.set_ylabel(self.metric_name, color='red')
                                #
                                # ax2.plot(t, self.history_scores, color='red')
                                #
                                # plt.locator_params(axis='y', nbins=40)
                                #
                                # fig.tight_layout()
                                #
                                # plt.show()

                if self.loss_graphic:
                    # pass
                    # if you have notebook (jupyter, kaggle notebook, colab notebook etc.) comment this block of code and uncomment 'pass'
                    fig, ax1 = plt.subplots(figsize=(9, 8))

                    clear_output(True)

                    ax1.set_xlabel('iters')

                    ax1.set_ylabel('Loss', color='blue')

                    t = np.arange(len(self.history_losses))

                    ax1.plot(t, self.history_losses)

                    ax2 = ax1.twinx()

                    ax2.set_ylabel(self.metric_name, color='red')

                    ax2.plot(t, self.history_scores, color='red')

                    plt.locator_params(axis='y', nbins=40)

                    fig.tight_layout()

                    plt.show()



            elif self.purpose == 'Regression':

                # to be continued

                pass

            else:
                raise Exception("Neural network purpose is not specified or unknown")

        except:

            raise Exception("Failed to train")

    def predict(self, x_test):

        self.hidden_outputs_activation = []

        result = x_test @ self.weights[0] + self.biases[0]

        self.hidden_outputs_activation.append(self.activation_func(result))

        for i in range(len(self.neurons)):

            result = self.hidden_outputs_activation[-1] @ self.weights[i + 1] + self.biases[i + 1]

            self.hidden_outputs_activation.append(self.activation_func(result))

        output = functions.softmax(result)

        return output

    def save(self, path = 'model_params.npy'):
        with open(path, 'wb') as f:
            for i in range(len(self.weights) - 1):
                np.save(f, self.weights[i])
                np.save(f, self.biases[i])
                np.save(f, self.last_grad_w[i])
                np.save(f, self.last_grad_b[i])
            np.save(f, self.weights[-1])
            np.save(f, self.last_grad_w[-1])

    def load(self, path = 'model_params.npy'):
        with open(path, 'rb') as f:
            for i in range(len(self.neurons) + 1):
                self.weights[i] = np.load(f)
                self.biases[i] = np.load(f)
                self.last_grad_w[i] = np.load(f)
                self.last_grad_b[i] = np.load(f)
            self.weights[-1] = np.load(f)
            self.last_grad_w[-1] = np.load(f)
            self.trained = True



