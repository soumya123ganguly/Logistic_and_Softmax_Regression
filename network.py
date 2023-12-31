import numpy as np
import data
import time
import math
# import tqdm

"""
NOTE
----
Start by implementing your methods in non-vectorized format - use loops and other basic programming constructs.
Once you're sure everything works, use NumPy's vector operations (dot products, etc.) to speed up your network.
"""

def sigmoid(a):
    """
    Compute the sigmoid function.

    f(x) = 1 / (1 + e ^ (-x))

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying sigmoid (z from the slides).
    """
    return 1/(1+np.exp(-a))

def softmax(a):
    """
    Compute the softmax function.

    f(x) = (e^x) / Σ (e^x)

    Parameters
    ----------
    a
        The internal value while a pattern goes through the network
    Returns
    -------
    float
       Value after applying softmax (z from the slides).
    """
    smax = np.exp(a)/np.sum(np.exp(a), axis=1)[:, None]
    # Sanity check to verify that all the softmax values in a column sum to 1.
    # assert(np.logical_and(np.sum(smax, axis=1) < 1.1, np.sum(smax, axis=1) > 0.9).all())
    return smax

def binary_cross_entropy(y, t):
    """
    Compute binary cross entropy.

    L(x) = t*ln(y) + (1-t)*ln(1-y)

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        binary cross entropy loss value according to above definition
    """
    return -(t*np.log(y+1e-15)+(1-t)*np.log(1-y+1e-15))

def multiclass_cross_entropy(y, t):
    """
    Compute multiclass cross entropy.

    L(x) = - Σ (t*ln(y))

    Parameters
    ----------
    y
        The network's predictions
    t
        The corresponding targets
    Returns
    -------
    float 
        multiclass cross entropy loss value according to above definition
    """
    return -np.sum(t*np.log(y+1e-15), axis=1)

class Network:
    def __init__(self, hyperparameters, activation, loss, out_dim):
        """
        Perform required setup for the network.

        Initialize the weight matrix, set the activation function, save hyperparameters.

        You may want to create arrays to save the loss values during training.

        Parameters
        ----------
        hyperparameters
            A Namespace object from `argparse` containing the hyperparameters
        activation
            The non-linear activation function to use for the network
        loss
            The loss function to use while training and testing
        """
        self.hyperparameters = hyperparameters
        self.activation = activation
        self.loss = loss

        self.weights = np.zeros((self.hyperparameters.p+1, out_dim))

    def forward(self, X):
        """
        Apply the model to the given patterns

        Use `self.weights` and `self.activation` to compute the network's output

        f(x) = σ(w*x)
            where
                σ = non-linear activation function
                w = weight matrix

        Make sure you are using matrix multiplication when you vectorize your code!

        Parameters
        ----------
        X
            Patterns to create outputs for
        """
        return self.activation(X.dot(self.weights))

    def __call__(self, X):
        return self.forward(X)

    def train(self, minibatch):
        """
        Train the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` and the gradient defined in the slides to update the network.

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
        tuple containing:
            average loss over minibatch
            accuracy over minibatch
        """
        # get the hyperparameters
        lr = self.hyperparameters.learning_rate
        bs = self.hyperparameters.batch_size
        X, y = minibatch
        # Compute the forward propogation of X.
        p = self.forward(X)
        # Compute the mean loss
        avg_loss = self.loss(p, y).mean()
        # Update the gradient weights
        self.weights += lr*X.T.dot(y-p)/bs
        # Used for computing binary classification accuracy
        #pred = np.where(p > 0.5, 1, 0)
        # Used for computing multiclass classification accuracy
        y = data.onehot_decode(y)
        pred = np.argmax(p, axis=1)+1
        # Compute the mean accuracy
        avg_acc = np.where(y == pred, 1, 0).mean()
        return avg_loss, avg_acc

    def test(self, minibatch):
        """
        Test the network on the given minibatch

        Use `self.weights` and `self.activation` to compute the network's output
        Use `self.loss` to compute the loss.
        Do NOT update the weights in this method!

        Parameters
        ----------
        minibatch
            The minibatch to iterate over

        Returns
        -------
            tuple containing:
                average loss over minibatch
                accuracy over minibatch
        """
        X, y = minibatch
        # Compute the forward propogation of X.
        p = self.forward(X)
        # Compute the mean loss
        avg_loss = self.loss(p, y).mean()
        # Used for computing binary classification accuracy
        #pred = np.where(p > 0.5, 1, 0)
        # Used for computing multiclass classification accuracy
        y = data.onehot_decode(y)
        pred = np.argmax(p, axis=1)+1
        # Compute the mean accuracy
        avg_acc = np.where(y == pred, 1, 0).mean()
        return avg_loss, avg_acc