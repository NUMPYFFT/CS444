"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        optimizer: str
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes  #len(hidden_sizes) = num_layers - 1
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # TODO: implement me
        return X @ W + b

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        # TODO: implement me
        return X * (X > 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        # TODO: implement me
        return X > 0

    def sigmoid(self, X: np.ndarray) -> np.ndarray:
        # TODO ensure that this is numerically stable
        return np.where(X < 0, np.exp(X) / (1 + np.exp(X)), 1 / (1 + np.exp(-X)))
    
    def sigmoid_grad(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        # TODO implement this
        return np.sum((y - p) ** 2, axis=1) / p.shape[1]   # take average over all classes
        
    def mse_grad(self, y: np.ndarray, p: np.ndarray)  -> np.ndarray:
        return -2 * (y - p) / p.shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.
        
        layer_output = X
        self.outputs[0] = X
        self.linear_out = {}
        
        for i in range(1, self.num_layers + 1):
            layer_output = self.linear(self.params["W" + str(i)] , layer_output , self.params["b" + str(i)])
            self.linear_out[i] = layer_output
            if i != self.num_layers:
                layer_output = self.relu(layer_output)
            else:
                layer_output = self.sigmoid(layer_output) # prediction
            self.outputs[i] = layer_output

        return layer_output # last layer output
        
        
    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.local_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        
        # x -> activation layer -> z
        
        dloss = self.mse_grad(y, self.outputs[self.num_layers])
        dact = self.sigmoid_grad(self.linear_out[self.num_layers])
        dl_dz = dloss * dact

        local_grad = self.outputs[self.num_layers - 1].T
        self.gradients["W" + str(self.num_layers)] = local_grad @ dl_dz
        self.gradients["b" + str(self.num_layers)] = np.sum(dl_dz, axis=0)

        dl_dx = dl_dz @ self.params["W" + str(self.num_layers)].T

        for i in range(self.num_layers - 1, 0, -1):
            dact = self.relu_grad(self.linear_out[i])
            dl_dz = dl_dx * dact
            local_grad = self.outputs[i - 1].T

            self.gradients["W" + str(i)] = local_grad @ dl_dz
            self.gradients["b" + str(i)] = np.sum(dl_dz, axis=0)
            dl_dx = dl_dz @ self.params["W" + str(i)].T
        
        loss = np.mean(self.mse(y, self.outputs[self.num_layers]))   # take the mean of losses over one batch, use this for neural_network.ipynb to be consistent
        # loss = np.sum(self.mse(y, self.outputs[self.num_layers]))   # take the sum of losses over one batch, use this for develop_neural_network.ipynb
        
        return loss
        
    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        # print("shape...", self.params['b1'].shape, self.gradients['b1'].shape)
        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)].reshape(-1)

        elif opt == "Adam":
            self.m = {key: 0 for key in self.gradients.keys()}
            self.v = {key: 0 for key in self.gradients.keys()}
            for i in range(1, self.num_layers + 1):
                
                self.m["W" + str(i)] = self.m["W" + str(i)] * b1 + (1 - b1) * self.gradients["W" + str(i)]
                self.v["W" + str(i)] = self.v["W" + str(i)] * b2 + (1 - b2) * self.gradients["W" + str(i)] ** 2
                self.m["b" + str(i)] = self.m["b" + str(i)] * b1 + (1 - b1) * self.gradients["b" + str(i)]
                self.v["b" + str(i)] = self.v["b" + str(i)] * b2 + (1 - b2) * self.gradients["b" + str(i)] ** 2
                self.params["W" + str(i)] -= lr * (self.m["W" + str(i)] / (1 - b1 ** i)) / (np.sqrt(self.v["W" + str(i)] / (1 - b2 ** i)) + eps)
                self.params["b" + str(i)] -= lr * (self.m["b" + str(i)] / (1 - b1 ** i)) / (np.sqrt(self.v["b" + str(i)] / (1 - b2 ** i)) + eps)

        
