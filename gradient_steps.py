import numpy as np


def gradient_descent(self, output, y_train):
    if self == None:
        raise Exception("No object in gradient descent")
    else:
        changes_w = []
        changes_b = []

        art = self.loss_derivative(output, y_train)

        for i in range(self.n_layers + 1):
            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(self.hidden_outputs_no_activation[-1 - i])

            changes_w.append(self.alpha * ((self.hidden_outputs_activation[-2 - i]).T @ art) / y_train.shape[0])
            changes_b.append(self.alpha * (np.sum(art) / y_train.shape[0]))

        for i in range(len(changes_w)):
            self.weights[-2 - i] = self.weights[-2 - i] - changes_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] - changes_b[i]


def stochastic_gradient_descent(self, output, y_train):
    if self == None:
        raise Exception("No object in gradient descent")
    else:
        changes_w = []
        changes_b = []

        k = np.random.randint(0, y_train.shape[0])

        art = self.loss_derivative(output, y_train)[k].reshape(1, -1)

        for i in range(self.n_layers + 1):

            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(
                    self.hidden_outputs_no_activation[-1 - i][k].reshape(1, -1))

            changes_w.append(self.alpha * ((self.hidden_outputs_activation[-2 - i][k].reshape(1, -1)).T @ art))
            changes_b.append(self.alpha * (np.sum(art)))

        for i in range(len(changes_w)):
            self.weights[-2 - i] = self.weights[-2 - i] - changes_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] - changes_b[i]

def stochastic_average_gradient_descent(self, output, y_train):
    if self == None:
        raise Exception("No object in gradient descent")
    else:

        norm = y_train.shape[0]

        changes_w = []
        changes_b = []

        k = np.random.randint(0, y_train.shape[0], (1, np.random.randint(2, int(y_train.shape[0] / 2))))[0]

        art = self.loss_derivative(output, y_train)[k]

        for i in range(self.n_layers + 1):

            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(self.hidden_outputs_no_activation[-1 - i][k])

            changes_w.append((self.alpha * ((self.hidden_outputs_activation[-2 - i][k]).T @ art)) / k.shape[0])
            changes_b.append(self.alpha * (np.sum(art)) / k.shape[0])

        for i in range(len(changes_w)):
            self.weights[-2 - i] = self.weights[-2 - i] - changes_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] - changes_b[i]
