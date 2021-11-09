import Network as Net
import Layer as Nl
import numpy as np

epsilon = 0.001


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


def process_mini_batch(network: Net.Network, training_set, ms, vs, number_of_updates:int, alpha=0.6, beta1=0.9, beta2=0.999):
    mini_bath_errors = []
    mini_bath_gradients = []
    for data, label in training_set:
        network.process_input(data)
        errors, gradients = get_errors_and_gradients(network, label)
        mini_bath_errors.append(errors)
        mini_bath_gradients.append(gradients)

    for l, layer in enumerate(network.layers):
        mean_gradients_in_batch_in_layer = sum([el[l] for el in mini_bath_gradients]) / len(training_set)
        M = beta1 * ms[l] + (1 - beta1) * mean_gradients_in_batch_in_layer
        V = beta2 * vs[l] + (1 - beta2) * (mean_gradients_in_batch_in_layer * mean_gradients_in_batch_in_layer)
        M_corrected = M / (1 - beta1**number_of_updates)
        V_corrected = V / (1 - beta2**number_of_updates)

        change = alpha / (np.sqrt(V_corrected) + epsilon) * M_corrected
        layer.weights = layer.weights - change

        bias_change = alpha / len(training_set) * sum([el[l] for el in mini_bath_errors])
        layer.bias = layer.bias - bias_change

        ms[l] = M
        vs[l] = V


def epoch(network, training_set, batch_size, mini_batches_count, ms, vs, number_of_updates:int, alpha=0.6, beta1=0.9, beta2=0.999):
    index = 0
    last_index = min(batch_size, len(training_set))
    for i in range(mini_batches_count):
        number_of_updates = number_of_updates + 1
        process_mini_batch(network, training_set[index:last_index], ms, vs, number_of_updates, alpha, beta1, beta2)
        index = last_index
        last_index = min(last_index + batch_size, len(training_set))
    return number_of_updates


def train_network_adam(network: Net.Network, training_set, validation_set, batch_size, iterations=8, alpha=0.6, beta1=0.9, beta2=0.999):
    mini_batches_count = -(len(training_set) // -batch_size)
    errors = []
    number_of_updates = 1
    ms, vs = create_zeros_ms_and_vs(network)
    for i in range(iterations):
        number_of_updates = epoch(network, training_set, batch_size, mini_batches_count, ms, vs, number_of_updates, alpha, beta1, beta2)
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


def create_zeros_ms_and_vs(network: Net.Network):
    ms = []
    vs = []
    for layer in network.layers:
        ms.append(np.zeros(shape=layer.weights.shape))
        vs.append(np.zeros(shape=layer.weights.shape))
    return ms, vs


def Test():
    net = Net.Network(6, [5, 3], [Nl.ReLU, Nl.ReLU], [Nl.ReLu_derivative, Nl.ReLu_derivative], 2)
    data1 = np.array([0, 1, 0, 1, 0, 1]).reshape((6, 1))
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
    train_network_adam(net, training_set, training_set, 5, 1)
    get_result(np.array([2, 3, 1, 0]).reshape((4, 1)))
    #epoch(net, training_set)


if __name__ == '__main__':
    Test()


