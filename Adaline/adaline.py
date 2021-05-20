import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Adaline:
    def transmute_targets(self, targets_train, targets_test):
        index = 0
        for target in enumerate(targets_train):
            if target == 0:
                targets_train[index] = -1
            index += 1
        index = 0
        for target in enumerate(targets_test):
            if target == 0:
                targets_test[index] = -1
            index += 1
        return targets_train, targets_test

    def sign_function(self, weighted_sum):
        if weighted_sum < 0:
            return -1
        else:
            return 1

    def output(self, weights, pattern):
        return np.dot(weights, pattern)

    def adjust_weights(self, weights, target, pattern, learning_rate, prediction):
        for i in range(len(weights)):
            inc = learning_rate * (target - prediction) * pattern[i]
            weights[i] += inc
        return weights

    def guess(self, weights, pattern):
        return self.sign_function(self.output(weights, pattern))

    def train(self, max_epochs, patterns, targets, learning_rate, min_mse):
        epoch = 0
        mse = 100
        weights = np.zeros(len(patterns[0]))
        while epoch < max_epochs and mse > min_mse:
            mse = 0
            for i, pattern in enumerate(patterns):
                current_output = self.output(weights, pattern)
                weights = self.adjust_weights(
                    weights, targets[i], pattern, learning_rate, current_output)
                mse += pow((targets[i] - current_output), 2)
            mse = mse / 2.0
            epoch += 1
        return weights

    def test(self, weights, patterns_test, targets_test):
        guesses = np.zeros(len(targets_test))
        for i in range(len(patterns_test)):
            guesses[i] = self.guess(weights, patterns_test[i])
        return guesses

    def plot_accuracy(self, targets, guesses):
        fig = plt.figure(2)
        fig.suptitle('Επιτυχείς/Ανεπιτυχείς κατανομές')
        for index in range(len(guesses)):
            plt.scatter(index, 'Chosen class' if targets[index] ==
                        1 else 'Other class', marker='o', color='blue', label='correct')
            plt.scatter(index, 'Chosen class' if guesses[index] ==
                        1 else 'Other class', marker='.', color='red', label='guesses')
        plt.xlabel("protypo")
        plt.ylabel("exodos (r) / stoxos (b)")
        plt.show()

    def cross_validation_test(self, patterns, targets, max_epochs, learning_rate, min_mse):
        fold_guesses = []
        fold_targets = []
        fold_patterns = []
        for i in range(9):
            patterns_train, patterns_test, targets_train, targets_test = train_test_split(
                patterns, targets, test_size=0.1)
            weights = self.train(max_epochs, patterns_train,
                                 targets_train, learning_rate, min_mse)
            guesses = self.test(weights, patterns_test, targets_test)
            fold_guesses.append(guesses)
            fold_targets.append(targets_test)
        self.plot_cross_validation(fold_guesses, fold_targets)

    def get_accuracy(self, guesses, targets):
        correct_answers = 0

        for i in range(len(guesses)):
            if guesses[i] == targets[i]:
                correct_answers += 1
        return int((correct_answers / len(targets)) * 100)

    def plot_cross_validation(self, fold_guesses, fold_targets):
        fig, axs = plt.subplots(3, 3)
        for i in range(3):
            for j in range(len(fold_guesses[i])):
                accuracy = self.get_accuracy(fold_guesses[i], fold_targets[i])
                axs[0, i].set_title(f"Accuracy: {accuracy}%")
                axs[0, i].scatter(j, np.array(fold_guesses)[
                                  i][j], c='tab:blue', marker='o', label='guesses')
                axs[0, i].scatter(j, np.array(fold_targets)[
                                  i][j], c='tab:red', marker='.', label='correct')
        for i in range(3):
            for j in range(len(fold_guesses[i])):
                accuracy = self.get_accuracy(fold_guesses[i], fold_targets[i])
                axs[1, i].set_title(f"Accuracy: {accuracy}%")
                axs[1, i].scatter(j, np.array(fold_guesses)[
                                  i+3][j], c='tab:blue', marker='o')
                axs[1, i].scatter(j, np.array(fold_targets)[
                                  i+3][j], c='tab:red', marker='.')
        for i in range(3):
            for j in range(len(fold_guesses[i])):
                accuracy = self.get_accuracy(fold_guesses[i], fold_targets[i])
                axs[2, i].set_title(f"Accuracy: {accuracy}%")
                axs[2, i].scatter(j, np.array(fold_guesses)[
                                  i+6][j], c='tab:blue', marker='o')
                axs[2, i].scatter(j, np.array(fold_targets)[
                                  i+6][j], c='tab:red', marker='.')
        plt.legend(['Guesses', 'Correct'], bbox_to_anchor=(
            1, 1), bbox_transform=plt.gcf().transFigure)
        fig.suptitle(
            'Επιτυχείς/ανεπιτυχείς κατανομές (9-fold cross-validation)')
        plt.show()
