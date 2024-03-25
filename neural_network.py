import functions
import metrics
import gradient_steps
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

class NeuralNetwork():

    def __init__(self, n_inputs, neurons, n_outputs, purpose='classification', batches=False):

        self.n_inputs = n_inputs

        self.neurons = neurons

        self.purpose = purpose

        self.n_outputs = n_outputs

        self.trained = False

        self.history_scores = []

        self.history_losses = []

    def prepare(self, gradient_method='gd', activation_func='sigmoid', alpha=0.1, seed=None,
                loss_function='cross_entropy_loss', optimizer=False):

        np.random.seed(seed)

        self.alpha = alpha

        if activation_func == 'sigmoid':
            self.activation_func = functions.sigmoid
            self.derivative = functions.derivative_sigmoid

        elif activation_func == 'relu':
            self.activation_func = functions.relu
            self.derivative = functions.derivative_relu

        elif activation_func == 'leaky_relu':
            self.activation_func = functions.leaky_relu
            self.derivative = functions.derivative_leaky_relu

        elif activation_func == 'tanh':
            self.activation_func = functions.tanh
            self.derivative = functions.derivative_tanh

        else:
            raise Exception("Activation function is not specified or unknown")

        if loss_function == 'cross_entropy_loss':
            self.loss_func = functions.cross_entropy_loss
            self.loss_derivative = functions.cross_entropy_loss_derivative
        else:
            raise Exception("Loss function is not specified or unknown")

        if gradient_method == 'gd':
            self.gradient_method = gradient_steps.gradient_descent

        elif gradient_method == 'sgd':
            self.gradient_method = gradient_steps.stochastic_gradient_descent

        elif gradient_method == 'sagd':
            self.gradient_method = gradient_steps.stochastic_average_gradient_descent

        else:
            raise Exception("gradient descent method is not specified or unknown")

    def cosmetic(self, progress_bar=False, loss_display=False, loss_graphic = False, iterations=0):

        self.loss_display = loss_display
        self.loss_graphic = loss_graphic

        self.iterations = iterations

        if not progress_bar:
            def tqdm_False(x, **params__):
                return x

            self.tqdm = NeuralNetwork.tqdm_False
        else:
            self.tqdm = tqdm

    def train(self, batches, test_batches, n_epochs):
        try:
            if self.purpose == 'classification':

                # this block of code needs to find out if model is already trained before
                if not self.trained:
                    # if not trained, let's initialize new params of the model
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

                    ## it is for accelerated momentum
                    self.last_grad_w = [0] * len(self.weights)
                    self.last_grad_b = [0] * len(self.biases)
                    ##

                    self.trained = True

                for epoch in tqdm(range(n_epochs), position=0, leave=True):

                    for index, batch in enumerate(batches):

                        x_train, y_train = batch

                        norm = x_train.shape[0]

                        # collect outputs from different layers for back propagation
                        self.hidden_outputs_no_activation = []
                        self.hidden_outputs_activation = []

                        self.hidden_outputs_activation.append(x_train)
                        self.hidden_outputs_no_activation.append(x_train)

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

                        loss = self.loss_func(output, y_train)

                        self.gradient_method(self, output, y_train)

                        #if self.loss_display and epoch % int(self.n_epochs / self.iterations) == 0:
                        if self.loss_display and index % 100 == 0:
                            val_acc = []
                            for batch in test_batches:

                                x_train, y_train = batch
                                val_acc.append(metrics.accuracy(self.predict(x_train), np.int_(np.arange(0, 10) == y_train)))

                            print('For epoch number: {}, validation accuracy is: {}, loss is {}'.format(epoch, round(
                                np.mean(val_acc), 4), round(np.mean(loss), 4)))

                            # print('For iter number: {}, validation accuracy is: {}'.format(index, round(
                            #      np.mean(val_acc), 4)))
                            self.history_losses.append(np.mean(loss))
                            self.history_scores.append(np.mean(val_acc))
                            if self.loss_graphic:

                                self.history_losses.append(np.mean(loss))
                                self.history_scores.append(np.mean(val_acc))

                                fig, ax1 = plt.subplots(figsize=(9, 8))

                                clear_output(True)

                                ax1.set_xlabel('iters')

                                ax1.set_ylabel('Loss', color='blue')

                                t = np.arange(len(self.history_losses))

                                ax1.plot(t, self.history_losses)

                                ax2 = ax1.twinx()

                                ax2.set_ylabel('Score', color='red')

                                ax2.plot(t, self.history_scores, color='red')

                                plt.locator_params(axis='y', nbins=40)

                                fig.tight_layout()

                                plt.show()



            elif self.purpose == 'Regression':

                # ADD SOMETHING

                pass

            else:
                raise Exception("Neural network purpose error")

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