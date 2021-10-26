import numpy as np
from scipy.special import softmax


def sigmoid(values):
    return 1 / (1 + np.exp(-values))


def sigmoid_derivative(values):
    return np.exp(-values) / ((np.exp(-values) + 1) ** 2)


def tanh(values):
    return np.tanh(values)


def tanh_derivative(values):
    return 1 - (values * values)


def ReLU(values):
    return values * (values > 0)


def ReLu_derivative(values):
    return values > 0


class Layer:
    def __init__(self, standard_deviation, input_size, size, activation_function=ReLU, derivative=ReLu_derivative):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.normal(scale=standard_deviation, size=(size, input_size))
        self.bias = np.random.normal(scale=standard_deviation, size=(size, 1))
        self.input = None
        self.z = None
        self.activation_function = activation_function
        self.derivative = derivative

    def calculate_output(self):
        self.z = self.weights @ self.input
        self.z = self.z + self.bias
        return self.activation_function(self.z)

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
    print(layer.calculate_output())
    soft = SoftMaxLayer()

    soft.input = layer.input
    print("input")
    print(soft.input)
    print("z")
    print(soft.z())

    test_to_der = np.array([1, 2, 3]).reshape((3, 1))
    print(tanh_derivative(test_to_der))
    print(sigmoid_derivative(test_to_der))
    print(ReLu_derivative(test_to_der))


if __name__ == "__main__":
    test()
