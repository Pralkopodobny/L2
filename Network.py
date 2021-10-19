import numpy as np

import Layer as Nl
import numpy as np


class Network:
    def __init__(self, input_size, sizes, activation_functions, standard_deviation=1):
        self.layers = []
        self.output = None
        prev_size = input_size
        for i, (size, func) in enumerate(zip(sizes, activation_functions)):
            self.layers.append(Nl.Layer(standard_deviation, input_size=prev_size, size=size, activation_function=func))
            prev_size = size
        self.layers.append(Nl.SoftMaxLayer())

    def process_input(self, values):
        self.layers[0].input = values
        prev_output = self.layers[0].z()
        for layer in self.layers[1:]:
            layer.input = prev_output
            prev_output = layer.z()
        self.output = prev_output

    def __str__(self):
        out = ""
        for layer in self.layers:
            out = out + layer.__str__()
        return out


def test():
    net = Network(3, [4, 2], [Nl.ReLU, Nl.ReLU])
    input = np.array([5, 1, 2]).reshape((3, 1))
    print(net.__str__())
    net.process_input(input)
    print(net.output)


if __name__ == "__main__":
    test()
