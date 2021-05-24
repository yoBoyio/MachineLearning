import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from adaline import Adaline
from sklearn.model_selection import train_test_split


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
    biases = np.ones((200, 0))
    return np.hstack((biases, patterns))


def plot_testing_data(patterns_train, patterns_test,):
    plt.scatter(range(len(patterns_train)), patterns_train, marker='o', color='b',
                label='training data')
    plt.scatter(range(len(patterns_test)), patterns_test, marker='.',
                color='r', label='testing data')
    plt.legend()
    plt.title('Κατανομή δεδομένων εκπαίδευσης/ελέγχου')
    plt.show()


def adaline_implementation(targets_train, targets_test, patterns_train, patterns_test,plot):
    a = Adaline()
    atargets_train, atargets_test = a.transmute_targets(
        targets_train, targets_test)
    max_epochs = int(input('Μέγιστος αριθμός εποχών: '))
    learning_rate = float(input('Ρυθμός εκμάθησης: '))
    min_mse = float(input('Ελάχιστο σφάλμα: '))
    weights = a.train(max_epochs, patterns_train,
                      atargets_train, learning_rate, min_mse,plot)
    if plot == False:
        guesses = a.test(weights, patterns_test, atargets_test)
        a.plot_accuracy(atargets_test, guesses)
    # a.cross_validation_test(patterns_train, atargets_train,
                            # max_epochs, learning_rate, min_mse)
