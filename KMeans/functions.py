import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline


def plot_data(targets):
    fig, ax = plt.subplots()
    ax.scatter(range(len(targets)), targets,  c='tab:blue',
               marker='o', label='Data')

    # ax.scatter(data[-50:, 0], data[-50:, 2], c='tab:green',
    #            marker='v', label='Iris-virginica')
    ax.legend(loc='lower right')
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('Κατανομή δεδομένων')
    plt.show()


def add_biases(patterns):
    biases = np.ones((len(patterns), 0))
    return np.hstack((biases, patterns))


def plot_testing_data(patterns_train, patterns_test,):
    plt.scatter(range(len(patterns_train)), patterns_train, marker='o', color='b',
                label='training data')
    plt.scatter(range(len(patterns_test)), patterns_test, marker='.',
                color='r', label='testing data')
    plt.legend()
    plt.title('Κατανομή δεδομένων εκπαίδευσης/ελέγχου')
    plt.show()


def plot_testing_data_iris(patterns_train, patterns_test,):
    plt.scatter(np.array(patterns_train)[:, 1], np.array(
        patterns_train)[:, 3], c='tab:blue', label='training data')
    plt.scatter(np.array(patterns_test)[:, 1], np.array(
        patterns_test)[:, 3], c='tab:red', label='testing data')
    plt.legend()
    plt.title('Κατανομή δεδομένων εκπαίδευσης/ελέγχου')
    plt.show()


def adaline_implementation(targets_train, targets_test, patterns_train,
                           patterns_test, plot, d3):
    a = Adaline()
    max_epochs = int(input('Μέγιστος αριθμός εποχών: '))
    learning_rate = float(input('Ρυθμός εκμάθησης: '))
    min_mse = float(input('Ελάχιστο σφάλμα: '))
    weights = a.train(max_epochs, patterns_train,
                      targets_train, learning_rate, min_mse, plot, d3)
    # if plot == False:
    guesses = a.test(weights, patterns_test, targets_test)
    a.plot_accuracy(targets_test, guesses)
    # a.cross_validation_test(patterns_train, atargets_train,
    # max_epochs, learning_rate, min_mse)
