import numpy as np

def gradient_descent(self, output, y_train):

    if self == None:
        raise Exception("No object in gradient descent")
    else:
        changes_w = []
        changes_b = []

        '''
        art - variable that stores gradient for every layer (step)
        it makes chain rule algorithm more straightforward to understand and implement
        you can easily check validity by calculating gradient descent on a piece of paper in a general form.
        '''

        art = self.loss_derivative(output, y_train)

        for i in range(len(self.neurons) + 1):
            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(self.hidden_outputs_no_activation[-1 - i])

            changes_w.append(self.alpha * ((self.hidden_outputs_activation[-2 - i]).T @ art) / y_train.shape[0])
            changes_b.append(self.alpha * (np.sum(art) / y_train.shape[0]))

        for i in range(len(changes_w)):

            # this is done for gradient descent algorithm called: "momentum"
            self.last_grad_w[i] = - changes_w[i] + self.momentum * self.last_grad_w[i]
            self.last_grad_b[i] = - changes_b[i] + self.momentum * self.last_grad_b[i]


            # updating weights and biases
            self.weights[-2 - i] = self.weights[-2 - i] + self.last_grad_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] + self.last_grad_b[i]

        try:
            if (np.isnan(self.weights[-2]).sum()) > 0:
                raise NaNException

        # this exception occurs if the training process has been broken down and the model is no longer being trained
        except NaNException:
            print("Training process failed, please decrease alpha parameter!")
            raise NaNException



def stochastic_gradient_descent(self, output, y_train):
    if self == None:
        raise Exception("No object in gradient descent")
    else:
        changes_w = []
        changes_b = []

        k = np.random.randint(0, y_train.shape[0])

        art = self.loss_derivative(output, y_train)[k].reshape(1, -1)

        for i in range(len(self.neurons) + 1):

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

        try:
            if (np.isnan(self.weights[-2]).sum()) > 0:
                raise NaNException


        except NaNException:
            print("Training process failed, please decrease alpha parameter!")
            raise NaNException


def stochastic_average_gradient_descent(self, output, y_train):
    if self == None:
        raise Exception("No object in gradient descent")
    else:

        norm = y_train.shape[0]

        changes_w = []
        changes_b = []

        k = np.random.randint(0, y_train.shape[0], (1, np.random.randint(2, int(y_train.shape[0] / 2))))[0]

        art = self.loss_derivative(output, y_train)[k]

        for i in range(len(self.neurons) + 1):

            if i == 0:
                art = (art @ (self.weights[-1 - i]).T)
            else:
                art = (art @ (self.weights[-1 - i]).T) * self.derivative(self.hidden_outputs_no_activation[-1 - i][k])

            changes_w.append((self.alpha * ((self.hidden_outputs_activation[-2 - i][k]).T @ art)) / k.shape[0])
            changes_b.append(self.alpha * (np.sum(art)) / k.shape[0])

        for i in range(len(changes_w)):
            self.weights[-2 - i] = self.weights[-2 - i] - changes_w[i]
            self.biases[-1 - i] = self.biases[-1 - i] - changes_b[i]

        try:
            if (np.isnan(self.weights[-2]).sum()) > 0:
                raise NaNException

        except NaNException:
            print("Training process failed, please decrease alpha parameter!")
            raise NaNException

class NaNException(Exception):
    "Training process failed, please decrease alpha parameter!"
    pass