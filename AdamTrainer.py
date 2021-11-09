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


def process_mini_batch(network: Net.Network, training_set, alpha=0.6, beta1=0.9, beta2=0.999):
    mini_bath_errors = []
    mini_bath_gradients = []
    for data, label in training_set:
        network.process_input(data)
        errors, gradients = get_errors_and_gradients(network, label)
        mini_bath_errors.append(errors)
        mini_bath_gradients.append(gradients)

    for layer_number, layer in enumerate(network.layers):
        layer.weights = layer.weights - alpha / len(training_set) * sum([el[layer_number] for el in mini_bath_gradients])
        layer.bia = layer.bias - alpha / len(training_set) * sum([el[layer_number] for el in mini_bath_errors])


def epoch(network, training_set, batch_size, mini_batches_count, alpha=0.6, beta1=0.9, beta2=0.999):
    index = 0
    last_index = min(batch_size, len(training_set))
    for i in range(mini_batches_count):
        process_mini_batch(network, training_set[index:last_index], alpha)
        index = last_index
        last_index = min(last_index + batch_size, len(training_set))


def train_network(network: Net.Network, training_set, validation_set, batch_size, iterations=8, alpha=0.6, beta1=0.9, beta2=0.999):
    mini_batches_count = -(len(training_set) // -batch_size)
    errors = []
    for i in range(iterations):
        epoch(network, training_set, batch_size, mini_batches_count, alpha)
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