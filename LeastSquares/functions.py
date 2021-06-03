from LeastSquares import LeastSquares
import matplotlib.pyplot as plt

import numpy as np


def least_square_implementation(xtrain, xtest, ttrain, ttest):

    # Init class
    l = LeastSquares()
    # Get correct weights
    a_train_target, a_test_target = l.transmute_targets(ttrain, ttest)
    weights = l.get_weights(xtrain, a_train_target)
    # Get predictions
    guesses = l.test(weights, xtest, a_test_target)
    # Plot results
    # guesses = l.guess(weights, xtest, a_test_target)
    show_guessing(a_test_target, guesses)
    print(guesses)


def add_biases(patterns):
    biases = np.ones((len(patterns), 0))
    return np.hstack((biases, patterns))


def show_guessing(ttrain, guesses):
    # Init 2 subplots
    fig, axs = plt.subplots(1)
    # Add title
    fig.suptitle('Correct results vs guesses')
    # For each item , add it to plot
    for index in range(len(guesses)):
        axs.scatter(index, 'Choosen class' if ttrain[index] ==
                    1 else 'Other class', marker='o', color='blue', label='correct')
        axs.scatter(index, 'Choosen class' if guesses[index] ==
                    1 else 'Other class', marker='.', color='red', label='guesses')
    # Show plot
    plt.show()
