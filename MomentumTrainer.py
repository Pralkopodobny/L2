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
        error = np.transpose(deeper_layer.weights) @ deeper_error * layer.derivative(layer.z)

        errors.append(error)
        gradients.append(error @ np.transpose(layer.input))

        deeper_layer = layer
        deeper_error = error
    return errors[::-1], gradients[::-1]


def epoch(network: Net.Network, training_set, previous_momentum, alpha=0.6, momentum_factor=0.9):
    mini_bath_errors = []
    mini_bath_gradients = []
    for data, label in training_set:
        network.process_input(data)
        errors, gradients = get_errors_and_gradients(network, label)
        mini_bath_errors.append(errors)
        mini_bath_gradients.append(gradients)

    for l, layer in enumerate(network.layers):
        change = previous_momentum[0][l] * momentum_factor + alpha / len(training_set) * sum([el[l] for el in mini_bath_gradients])
        previous_momentum[0][l] = change
        layer.weights = layer.weights - change
        bias_change = previous_momentum[1][l] * momentum_factor + alpha / len(training_set) * sum([el[l] for el in mini_bath_errors])
        previous_momentum[1][l] = bias_change
        layer.bia = layer.bias - bias_change


def iteration(network, training_set, batch_size, mini_batches_count, previous_momentum, alpha=0.6, momentum_factor=0.9):
    index = 0
    last_index = min(batch_size, len(training_set))
    for i in range(mini_batches_count):
        epoch(network, training_set[index:last_index], previous_momentum, alpha, momentum_factor)
        index = last_index
        last_index = min(last_index + batch_size, len(training_set))


def train_network_momentum(network: Net.Network, training_set, validation_set, batch_size, iterations=8, alpha=0.6, momentum_factor=0.9):
    mini_batches_count = -(len(training_set) // -batch_size)
    errors = []
    previous_momentum = create_zero_momentum(network.layers)
    for i in range(iterations):
        iteration(network, training_set, batch_size, mini_batches_count, previous_momentum, alpha, momentum_factor)
        val_err = error(network, validation_set)
        errors.append(val_err)
        print(val_err)
    return errors


def acc(network, data_set):
    correct = 0
    for image, label in data_set:
        network.process_input(image)
        correct = correct + (label[get_result(network.output)] == 1)
    return correct / len(data_set)


def error(network, data_set):
    return 1 - acc(network, data_set)


def get_result(input):
    return np.argmax(input)


def create_zero_momentum(layers: Nl.Layer):
    momentum = [[], []]
    for layer in layers:
        momentum[0].append(np.zeros(shape=layer.weights.shape))
        momentum[1].append(np.zeros(shape=layer.bias.shape))
    return momentum
