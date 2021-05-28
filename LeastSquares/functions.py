from LeastSquares import LeastSquares
import matplotlib.pyplot as plt


def least_square_implementation(xtrain, xtest, ttrain, ttest):

    # Init class
    l = LeastSquares()
    # Get correct weights
    weights = l.get_weights(xtrain, ttrain)
    # Get predictions
    guesses = l.get_predictions(xtest, weights)
    # Plot results
    show_guessing(ttest, guesses)
    # Test for 9 folds
    # print('Now testing with 9 folds')
    # fold_guesses = []
    # fold_t = []
    # for _ in range(9):
    #     # Get sets
    #     xtrain, xtest, ttrain, ttest = d.return_splits(table_t, table_x)
    #     l2 = LeastSquares()
    #     weights = l2.get_weights(xtrain, ttrain)
    #     guesses = l2.get_predictions(xtest, weights)
    #     # Append results
    #     fold_guesses.append(guesses)
    #     fold_t.append(ttest)
    # # Plot results
    # # fold_results(fold_guesses, fold_t)


def show_guessing(self, ttrain, guesses):
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


def fold_results(self, fold_guesses, fold_targets):
    # Init 9 subplots
    fig, axs = plt.subplots(3, 3)
    # For first 3 show items
    for i in range(3):
        for j in range(len(fold_guesses[i])):
            axs[0, i].scatter(j, np.array(fold_guesses)[i]
                              [j], c='tab:blue', marker='o')
            axs[0, i].scatter(j, np.array(fold_targets)[i]
                              [j], c='tab:red', marker='.')
    # For 3 through 6 show items
    for i in range(3):
        for j in range(len(fold_guesses[i])):
            axs[1, i].scatter(j, np.array(fold_guesses)[i+3]
                              [j], c='tab:blue', marker='o')
            axs[1, i].scatter(j, np.array(fold_targets)[i+3]
                              [j], c='tab:red', marker='.')
    # For 6 to 9 show items
    for i in range(3):
        for j in range(len(fold_guesses[i])):
            axs[2, i].scatter(j, np.array(fold_guesses)[i+6]
                              [j], c='tab:blue', marker='o')
            axs[2, i].scatter(j, np.array(fold_targets)[i+6]
                              [j], c='tab:red', marker='.')
    plt.legend(['Guesses', 'Correct class'])
    plt.show()
