import numpy as np
np.random.seed(0)

class ReLU():

    def __init__(self):
        pass

    def calculate(X):
        return np.maximum(X, 0)

    def derivate(X):
        return X > 0


class Softmax():

    def __init__(self):
        pass

    def calculate(X):
        return np.exp(X) / sum(np.exp(X))


class Input():

    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        self.n_features = n_features


class Dense():

    def __init__(self, n_layers, n_features, activation=ReLU):
        self.weight = np.random.rand(n_layers, n_features) - 0.5
        self.bias = np.random.rand(n_layers, 1) - 0.5 # scalar
        self.activation = activation # activation function

    def forward(self, A):
        self.Z = self.weight.dot(A) + self.bias
        self.A = self.activation.calculate(self.Z)
        return self.A
    
    def backward(self, X, next_layer):
        m = X.shape[1]
        self.dZ = next_layer.weight.T.dot(next_layer.dZ) * self.activation.derivate(self.Z)
        self.dW = 1 / m * self.dZ.dot(next_layer.A.T)
        self.db = 1 / m * np.sum(self.dZ)