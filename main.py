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
    network = Net.Network(input_size=784, sizes=[10, 8], activation_functions=[Nl.sigmoid, Nl.sigmoid],
                          derivatives=[Nl.sigmoid_derivative, Nl.sigmoid_derivative], output_size=10)
    print("zabawa rozpoczeta")
    for i in range(15):
        print(i)
        Tr.train_network(network, training_set, 70)

    sum = 0
    for image, label in training_set:
        network.process_input(image)
        sum = sum + (label[Tr.get_result(network.output)] == 1)

    print(sum / len(training_set))



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nyan()
    #prepare_training_set()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
