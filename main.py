import numpy as np
from mnist import MNIST
from matplotlib import pyplot as plt
import Network as Net
import Layer as Nl
import Trainer as Tr


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

    #print(training_set[0])

    return training_set


def nyan():
    training_set = prepare_training_set()
    network = Net.Network(input_size=784, sizes=[10, 8], activation_functions=[Nl.ReLU, Nl.ReLU],
                          derivatives=[Nl.ReLu_derivative, Nl.ReLu_derivative], output_size=10)
    print("zabawa rozpoczeta")
    for i in range(100):
        print(i)
        Tr.train_network(network, training_set, 70)
    image, label = training_set[0][0], training_set[0][1]
    network.process_input(image)

    print("wynik")
    print(network.output)
    print("prawda")
    print(label)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nyan()
    #prepare_training_set()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
