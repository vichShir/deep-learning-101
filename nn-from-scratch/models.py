import numpy as np

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_predictions(y):
    return np.argmax(y, 0)

def get_accuracy(predictions, Y):
    #print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

class Model():

    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def __forward(self, X):
        A = self.layers[0].forward(X) # get the output of the first layer
        for layer in self.layers[1:]:
            A = layer.forward(A)

    def __backward(self, X, y):
        m = X.shape[1]
        one_hot_Y = one_hot(y)

        # softmax
        softmax_layer = self.layers[-1]
        softmax_layer.dZ = softmax_layer.A - one_hot_Y
        softmax_layer.dW = 1 / m * softmax_layer.dZ.dot(self.layers[-2].A.T)
        softmax_layer.db = 1 / m * np.sum(softmax_layer.dZ)

        # middle dense layers
        for i in range(len(self.layers) - 1, 1, -1):
            self.layers[i-1].backward(X, self.layers[i])

        # last dense layer
        last_layer = self.layers[0]
        last_layer.dZ = self.layers[1].weight.T.dot(self.layers[1].dZ) * last_layer.activation.derivate(last_layer.Z)
        last_layer.dW = 1 / m * last_layer.dZ.dot(X.T)
        last_layer.db = 1 / m * np.sum(last_layer.dZ)

    def __update_params(self, lr):
        for i in range(len(self.layers)):
            self.layers[i].weight = self.layers[i].weight - lr * self.layers[i].dW
            self.layers[i].bias = self.layers[i].bias - lr * self.layers[i].db

    def gradient_descent(self, X, y, lr, epochs):
        for i in range(epochs):
            self.__forward(X)
            self.__backward(X, y)
            self.__update_params(lr)
            if i % 10 == 0:
                print("Iteration: ", i)
                predictions = get_predictions(self.layers[-1].A)
                print(get_accuracy(predictions, y))

    def predict(self, X):
        A = self.layers[0].forward(X) # get the output of the first layer
        for layer in self.layers[1:]:
            A = layer.forward(A)
        return get_predictions(A)