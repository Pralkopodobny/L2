import numpy as np
from mnist import MNIST
import Network as Net
import Layer as Nl
import Trainer as Tr
import MomentumTrainer as MomentumTr
import AdagradTrainer as AdagardTr
import AdadeltaTrainer as AdadeltaTr


def prepare_training_set():
    mndata = MNIST('./dataset')
    mndata.gz = True
    images, labels = mndata.load_training()
    labels_n_to_1 = []
    images_normalized = []

    labels = np.array(labels)
    images = np.array(images)
    images = images / 255
    for label in labels:
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l[label] = 1
        labels_n_to_1.append(np.array(l).reshape(10, 1))

    for image in images:
        images_normalized.append(image.reshape(784, 1))

    training_set = [list(a) for a in zip(images_normalized, labels_n_to_1)]
    return training_set


def prepare_validation_set():
    mndata = MNIST('./dataset')
    mndata.gz = True
    images, labels = mndata.load_testing()
    labels_n_to_1 = []
    images_normalized = []

    labels = np.array(labels)
    images = np.array(images)
    images = images / 255
    for label in labels:
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        l[label] = 1
        labels_n_to_1.append(np.array(l).reshape(10, 1))

    for image in images:
        images_normalized.append(image.reshape(784, 1))

    training_set = [list(a) for a in zip(images_normalized, labels_n_to_1)]

    return training_set


def early_test():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                          derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                          standard_deviation=0.3)
    Tr.early_return_training(network, training_set, validation_set, 600)


def test_alpha():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    iterations_count = 6

    def single_experiment(alpha):
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                              derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                              standard_deviation=0.3)
        print("start")
        ek = Tr.train_network(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)
        return ek

    def single_alpha_experiment(alpha):
        single_alpha_results = []
        for i in range(10):
            single_alpha_results.append(single_experiment(alpha))
        return np.array(single_alpha_results).mean(axis=1)

    alphas = [0.4, 0.5, 0.6, 0.7]
    results = []
    for alpha in alphas:
        results.append(single_experiment(alpha))

    def results_to_text(results, iterations):
        text = ""
        for iteration in range(iterations):
            text = text + f"{results[0][iteration]}"
            for result in results[1::]:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_mini_batch():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 10

    def single_experiment(batch_size):
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                              derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                              standard_deviation=0.3)
        return Tr.train_network(network, training_set, validation_set, batch_size=batch_size, alpha=alpha, iterations=iterations_count)

    def single_batch_experiment(batch_size):
        single_alpha_results = []
        for i in range(10):
            single_alpha_results.append(single_experiment(batch_size))
        return np.array(single_alpha_results).mean(axis=1)

    batch_sizes = [6000, 12000, 18000, 24000]
    results = []
    for batch_size in batch_sizes:
        results.append(single_experiment(batch_size))

    def results_to_text(results, iterations):
        text = ""
        for iteration in range(iterations):
            text = text + f"{iteration + 1};"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_standard_deviation():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment(standard_deviation):
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                              derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                              standard_deviation=standard_deviation)
        return Tr.train_network(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    def single_batch_experiment(standard_deviation):
        single_alpha_results = []
        for i in range(10):
            single_alpha_results.append(single_experiment(standard_deviation))
        return np.array(single_alpha_results).mean(axis=1)

    deviations = [0.1, 0.2, 0.3, 0.4]
    results = []
    for batch_size in deviations:
        results.append(single_experiment(batch_size))

    def results_to_text(results, iterations):
        text = "Pomiar;"
        for dev in deviations:
            text = text + f";odchylenie={dev}"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_activation_functions():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment(activation_functions, derivatives):
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=activation_functions,
                              derivatives=derivatives, output_size=10,
                              standard_deviation=0.3)
        return Tr.train_network(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    def single_batch_experiment(activation_functions, derivatives):
        single_alpha_results = []
        for i in range(10):
            single_alpha_results.append(single_experiment(activation_functions, derivatives))
        return np.array(single_alpha_results).mean(axis=1)

    act = [[Nl.ReLU, Nl.ReLU], [Nl.sigmoid, Nl.ReLU]]
    der = [[Nl.ReLu_derivative, Nl.ReLu_derivative], [Nl.sigmoid_derivative, Nl.ReLu_derivative]]
    results = []
    for activations, der in zip(act, der):
        results.append(single_experiment(activations, der))

    def results_to_text(results, iterations):
        text = "Pomiar;reLu-ReLU;Sigmoid-ReLU"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_layer_sizes():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment(sizes):
        network = Net.Network(input_size=784, sizes=sizes, activation_functions=[Nl.ReLU, Nl.ReLU],
                              derivatives=[Nl.ReLu_derivative, Nl.ReLu_derivative], output_size=10,
                              standard_deviation=0.3)
        return Tr.train_network(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    def single_batch_experiment(sizes):
        single_alpha_results = []
        for i in range(10):
            single_alpha_results.append(single_experiment(sizes))
        return np.array(single_alpha_results).mean(axis=1)

    layer_sizes = [[120, 40], [60, 40]]
    results = []
    for l_sizes in layer_sizes:
        results.append(single_experiment(l_sizes))

    def results_to_text(results, iterations):
        text = "Pomiar"
        for l_sizes in layer_sizes:
            text = text + f";{l_sizes}"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_momentum():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment():
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.ReLU, Nl.ReLU],
                              derivatives=[Nl.ReLu_derivative, Nl.ReLu_derivative], output_size=10,
                              standard_deviation=0.3)
        return MomentumTr.train_network_momentum(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count, momentum_factor=0.1)

    results = []
    results.append(single_experiment())

    def results_to_text(results, iterations):
        text = "Pomiar;Wynik"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_adagard():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment():
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                              derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                              standard_deviation=0.3)
        return AdagardTr.train_network_adagard(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    def single_experiment_relu():
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.ReLU, Nl.ReLU],
                              derivatives=[Nl.ReLu_derivative, Nl.ReLu_derivative], output_size=10,
                              standard_deviation=0.3)
        return AdagardTr.train_network_adagard(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    results = []
    results.append(single_experiment())

    def results_to_text(results, iterations):
        text = "Pomiar;Wynik"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def test_adadelta():
    training_set = prepare_training_set()
    validation_set = prepare_validation_set()
    alpha = 0.7
    iterations_count = 6

    def single_experiment():
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                              derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10,
                              standard_deviation=0.3)
        return AdadeltaTr.train_network_adadelta(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    def single_experiment_relu():
        network = Net.Network(input_size=784, sizes=[30, 14], activation_functions=[Nl.ReLU, Nl.ReLU],
                              derivatives=[Nl.ReLu_derivative, Nl.ReLu_derivative], output_size=10,
                              standard_deviation=0.3)
        return AdadeltaTr.train_network_adadelta(network, training_set, validation_set, batch_size=600, alpha=alpha, iterations=iterations_count)

    results = []
    results.append(single_experiment_relu())

    def results_to_text(results, iterations):
        text = "Pomiar;Wynik"
        for iteration in range(iterations):
            text = text + f"{iteration + 1}"
            for result in results:
                text = text + f";{result[iteration]}"
            text = text + "\n"
        return text
    return results_to_text(results, iterations_count)


def some_testing():
    a = np.array([[1, 2], [2, 3]])
    print(a[0][0])
    print(a[1::])
    print(np.mean(a, axis=1))

if __name__ == '__main__':
    #some_testing()
    #text = test_alpha()
    #text = test_standard_deviation()
    #text = test_activation_functions()
    #text = test_layer_sizes()
    #text = test_momentum()
    #text = test_adagard()
    text = test_adadelta()



    with open("Output.txt", "w") as text_file:
        text_file.write(text)




