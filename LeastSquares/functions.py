from LeastSquares import LeastSquares
import matplotlib.pyplot as plt


def least_square_implementation(xtrain, xtest, ttrain, ttest):

    # Init class
    l = LeastSquares()
    # Get correct weights
    a_train_target, a_test_target = l.transmute_targets(ttrain, ttest)
    weights = l.get_weights(xtrain, a_train_target)
    # Get predictions
    train = l.test(weights, xtest, a_test_target)
    # Plot results
    guesses = l.guess(weights, xtest)
    show_guessing(train, guesses)
    print(guesses)


def show_guessing(ttrain, guesses):
    # Init 2 subplots
    fig, axs = plt.subplots(2)
    # Add title
    fig.suptitle('Correct results vs guesses')
    # For each item , add it to plot
    for index in range(len(guesses)):
        axs[0].scatter(index, 'Choosen class' if ttrain[index] ==
                       1 else 'Other class', marker='o', color='blue', label='correct')
        axs[1].scatter(index, 'Choosen class' if guesses[index] ==
                       1 else 'Other class', marker='o', color='red', label='guesses')
    # Show plot
    plt.show()
