"""Perceptron model."""

import numpy as np
import copy
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay_rate: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay_rate = decay_rate

    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100

    def train(self, X_train: np.ndarray, y_train: np.ndarray, plot=True):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, d = X_train.shape
        accuracy = np.zeros(self.epochs)
        # self.w = np.random.rand(self.n_class, d+1)
        self.w = np.ones((self.n_class, d+1)) / d
        X_train = np.append(X_train, np.ones((N, 1)), axis=1)   # add a column ones for bias

        for i in range(self.epochs):
            for data, label in zip(X_train, y_train):
                true_score = np.dot(self.w[label], data)
                incorrect_classes = np.nonzero(np.arange(self.n_class) - label)[0]
                for c in incorrect_classes:
                    pred_score = np.dot(self.w[c], data)
                    if pred_score >= true_score:
                        self.w[c] = self.w[c] - self.lr * data
                        self.w[label] = self.w[label] + self.lr * data 
            self.lr = self.lr * (1 / (1 + self.decay_rate * self.epochs))
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
        X_test = np.append(X_test, np.ones((N, 1)), axis=1)   # add a column ones for bias
        scores = self.w @ X_test.T
        return np.argmax(scores, axis=0)
