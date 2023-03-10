"""Logistic regression model."""

import numpy as np
import copy

class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float, decay_rate: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold
        self.decay_rate = decay_rate
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implement me
        if x < 0:
            sigmoid = np.exp(x) / (1 + np.exp(x))
        else: 
            sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implement me
        N, d = X_train.shape 
        y_train_copy = copy.deepcopy(y_train)
        y_train_copy[np.where(y_train_copy == 0)] = -1   # labels now in {-1, 1}
        X_train = np.append(X_train, np.ones((N, 1)), axis=1)   # add a column of ones for bias
        self.w = np.zeros(d+1)
        
        for i in range(self.epochs):
            for data, label in zip(X_train, y_train_copy):
                self.w = self.w + self.lr * self.sigmoid(-label * np.dot(self.w, data)) * label * data
            self.lr = self.lr * (1 / (1 + self.decay_rate * self.epochs))
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
        X_test = np.append(X_test, np.ones((N, 1)), axis=1)   # add a column of ones for bias
        pred_label = (self.w @ X_test.T) > 0
        return pred_label
