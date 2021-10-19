import numpy as np
from mnist import MNIST
import Network as Net
import Layer as Nl

def print_hi(name):
    mndata = MNIST('./dataset')
    mndata.gz = True
    images, labels = mndata.load_training()
    array = np.array(images)
    network = Net.Network(784, [14, 10, 10], [Nl.tanh, Nl.tanh, Nl.tanh])
    network.process_input(array[0].reshape((784, 1)))
    print(network.output)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
