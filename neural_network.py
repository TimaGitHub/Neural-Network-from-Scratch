import functions
import metrics
import gradient_steps
from tqdm.auto import tqdm
import numpy as np

class NeuralNetwork():

    def __init__(self, n_layers, n_neurons, purpose='classification'):

        self.n_layers = n_layers

        self.n_neurons = n_neurons

        self.purpose = purpose

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

    def cosmetic(self, progress_bar=False, loss_display=False, iterations=0):

        self.loss_display = loss_display

        self.iterations = iterations

        if not progress_bar:
            def tqdm_False(x, **params__):
                return x

            self.tqdm = NeuralNetwork.tqdm_False
        else:
            self.tqdm = tqdm

    def train(self, x_train, y_train, n_epochs):
        try:
            self.n_epochs = n_epochs

            if self.purpose == 'classification':

                # this block of code needs to find out if model is already trained before
                if not self.trained:
                    # if not trained, let's initialize new params of the model
                    self.n_inputs = x_train.shape[1] # define the number of features
                    self.n_outputs = y_train.shape[1] #d efine the number of classes

                    self.weights = [np.random.uniform(- 0.5, 0.5, size=(self.n_inputs, self.n_neurons))]
                    self.biases = [np.random.uniform(- 0.5, 0.5, size=(1, self.n_neurons))]

                    for i in range(self.n_layers - 1):

                        self.weights.append(np.random.uniform(-0.5, 0.5, size=(self.n_neurons, self.n_neurons)))
                        self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, self.n_neurons)))

                    self.weights.append(np.random.uniform(-0.5, 0.5, size=(self.n_neurons, self.n_outputs)))

                    self.biases.append(np.random.uniform(-0.5, 0.5, size=(1, self.n_outputs)))

                    # identity matrix at the end needs for correct chain rule algorithm
                    self.weights.append(np.eye(self.n_outputs, self.n_outputs))

                for epoch in tqdm(range(self.n_epochs + 1), position=0, leave=True):

                    norm = x_train.shape[0]

                    # collect outputs from different layers for back propagation
                    self.hidden_outputs_no_activation = []
                    self.hidden_outputs_activation = []

                    self.hidden_outputs_activation.append(x_train)
                    self.hidden_outputs_no_activation.append(x_train)

                    result = x_train @ self.weights[0] + self.biases[0]

                    self.hidden_outputs_no_activation.append(result)
                    self.hidden_outputs_activation.append(self.activation_func(result))

                    for i in range(self.n_layers):

                        result = self.hidden_outputs_activation[-1] @ self.weights[i + 1] + self.biases[i + 1]

                        self.hidden_outputs_no_activation.append(result)

                        self.hidden_outputs_activation.append(self.activation_func(result))

                    # remove last result to apply softmax function
                    pop_garbage = self.hidden_outputs_activation.pop()

                    output = functions.softmax(result)

                    self.hidden_outputs_activation.append(output)

                    loss = self.loss_func(output, y_train)

                    self.gradient_method(self, output, y_train)

                    if self.loss_display and epoch % int(self.n_epochs / self.iterations) == 0:
                        print('For epoch number: {}, prediction accuracy is: {}'.format(epoch, round(
                            metrics.accuracy(output, y_train), 4)))



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

        for i in range(self.n_layers):

            result = self.hidden_outputs_activation[-1] @ self.weights[i + 1] + self.biases[i + 1]

            self.hidden_outputs_activation.append(self.activation_func(result))

        output = functions.softmax(result)

        return output