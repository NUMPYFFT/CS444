"""Support Vector Machine (SVM) model."""

import numpy as np
import copy
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, batch_size: int, decay_rate: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.batch_size = batch_size
        self.decay_rate = decay_rate
    
    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implement me
        N, d = X_train.shape
        gradients = np.zeros((self.n_class, d))   # (num of classes, batch size, feature size+1)
        for (data, label) in zip(X_train, y_train):                                 
            for c in range(self.n_class):
                if np.dot(self.w[label], data) - np.dot(self.w[c], data) < 1:                     
                    if c != label:
                        gradients[c] += data
                        gradients[label] -= data
        gradients += self.reg_const * self.w
        gradients /= N
        return gradients

    def train(self, X_train: np.ndarray, y_train: np.ndarray, plot=True):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, d = X_train.shape
        accuracy = np.zeros(self.epochs)
        # self.w = np.random.rand(self.n_class, d+1)
        self.w = np.ones((self.n_class, d+1)) / d
        X_train = np.append(X_train, np.ones((N, 1)), axis=1)   # add a column of ones for bias
        
        for i in range(self.epochs):
            sample_indices = np.random.choice(N, self.batch_size)
            X_batch = X_train[sample_indices]
            y_batch = y_train[sample_indices]
            self.w = self.w - self.lr * self.calc_gradient(X_batch, y_batch)
            self.lr = self.lr * (self.decay_rate / np.sqrt(self.batch_size))
            accuracy[i] = self.get_acc(np.argmax(self.w @ X_train.T, axis=0), y_train)

        if plot == True:
            plt.figure(figsize=(5,3))
            plt.plot(np.arange(self.epochs), accuracy)
            plt.title("training accuracy vs epochs")
            plt.xlabel("epochs")
            plt.ylabel("accuracy (%)")
            plt.grid()

        return

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        # TODO: implement me
        N, d = X_test.shape
        X_test = np.append(X_test, np.ones((N, 1)), axis=1)
        scores = self.w @ X_test.T
        return np.argmax(scores, axis=0)
        
        