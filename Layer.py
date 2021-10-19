import numpy as np
from scipy.special import softmax


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def tanh(values):
    return np.tanh(values)


def ReLU(values):
    return values * (values > 0)


class Layer:
    def __init__(self, standard_deviation, input_size, size, activation_function=ReLU):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.normal(scale=standard_deviation, size=(size, input_size))
        self.bias = np.random.normal(scale=standard_deviation, size=(size, 1))
        self.input = None
        self.activation_function = activation_function

    def z(self):
        temp = self.weights @ self.input
        temp = temp + self.bias
        return self.activation_function(temp)

    def __str__(self):
        return f"(input_size:{self.input_size}, size:{self.size})"


class SoftMaxLayer:
    def __init__(self):
        self.input = None

    def z(self):
        return softmax(self.input)


def test():
    layer = Layer(1, input_size=4, size=3)
    layer.input = np.array([1, 0, 1, 0.3]).reshape((4, 1))
    print(layer.weights)
    print(layer.input)
    print("bias")
    print(layer.bias)
    print("mult")
    print(layer.weights @ layer.input)
    print("z")
    print(layer.z())
    soft = SoftMaxLayer()

    soft.input = layer.input
    print("input")
    print(soft.input)
    print("z")
    print(soft.z())


if __name__ == "__main__":
    test()
