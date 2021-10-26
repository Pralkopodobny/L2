import Network as Net
import Layer as Nl
import numpy as np


def get_errors_and_gradients(network: Net.Network, expected_solution):
    layers = network.layers
    errors = []
    gradients = []

    last_layer_error = -(expected_solution - network.output)
    last_layer_gradient = last_layer_error @ np.transpose(layers[-1].input)

    errors.append(last_layer_error)
    gradients.append(last_layer_gradient)

    deeper_layer = layers[-1]
    deeper_error = last_layer_error

    for layer in layers[-2::-1]:  # reversed without last
        first = np.transpose(deeper_layer.weights)
        sec = deeper_error
        third = layer.derivative(layer.z)
        error = first @ sec * third
        #print(f"{error.shape}, {np.transpose(layer.input).shape}")
        gradient = error @ np.transpose(layer.input)

        errors.append(error)
        gradients.append(gradient)

        deeper_layer = layer
        deeper_error = error
    return errors[::-1], gradients[::-1]


def epoch(network: Net.Network, training_set, alpha=0.1):
    mini_bath_errors = []
    mini_bath_gradients = []
    for data, label in training_set:
        network.process_input(data)
        errors, gradients = get_errors_and_gradients(network, label)
        mini_bath_errors.append(errors)
        mini_bath_gradients.append(gradients)

    mini_bath_average_gradients = average(mini_bath_gradients)
    mini_bath_average_errors = average(mini_bath_errors)

    for layer, average_gradient, average_error in zip(network.layers, mini_bath_average_gradients,
                                                      mini_bath_average_errors):
        layer.weights = layer.weights - average_gradient * alpha
        layer.bias = layer.bias - average_error * alpha


def train_network(network, training_set, batch_size):
    mini_batches_count = -(len(training_set) // -batch_size)
    index = 0
    last_index = min(batch_size, len(training_set))
    for i in range(mini_batches_count):
        epoch(network, training_set[index:last_index])
        index = last_index
        last_index = min(last_index + batch_size, len(training_set))


def average(table):
    sums = table[0]
    averages = []
    for errors in table[1::]:
        for i, error in enumerate(errors):
            sums[i] = sums[i] + error
    for sum in sums:
        averages.append(sum / len(table))
    return averages


def Test():
    net = Net.Network(6, [5, 3], [Nl.ReLU, Nl.ReLU], [Nl.ReLu_derivative, Nl.ReLu_derivative], 2)
    data1 = np.array([0, 0, 0, 0, 0, 0]).reshape((6, 1))
    data2 = np.array([1, 1, 1, 1, 1, 1]).reshape((6, 1))
    data3 = np.array([1, 0, 0, 0, 1, 1]).reshape((6, 1))
    data4 = np.array([0.5, 0.5, 1, 1, 1, 1]).reshape((6, 1))
    data5 = np.array([0.5, 0.5, 0, 0, 0, 1]).reshape((6, 1))
    data = [data1, data2, data3, data4, data5]
    expected1 = np.array([0, 1]).reshape((2, 1))
    expected2 = np.array([1, 0]).reshape((2, 1))
    expected3 = np.array([1, 0]).reshape((2, 1))
    expected4 = np.array([0, 1]).reshape((2, 1))
    expected5 = np.array([0, 1]).reshape((2, 1))
    expected = [expected1, expected2, expected3, expected4, expected5]
    training_set = [list(a) for a in zip(data, expected)]
    train_network(net, training_set, 2)
    epoch(net, training_set)


if __name__ == '__main__':
    Test()
