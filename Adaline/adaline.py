import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class Adaline:
    def transmute_targets(self, targets_train, targets_test):
        index = 0
        for target in enumerate(targets_train):
            if target[1] == 0:
                # print(target[1])
                targets_train[index] = -1
            index += 1
        index = 0
        for target in enumerate(targets_test):
            if target[1] == 0:
                print(target[1])
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

    def train(self, max_epochs, patterns, targets, learning_rate,
              min_mse, plot=False, d3=False):
        epoch = 0
        mse = 100
        if plot:
            fig, axes = plt.subplots(1, 4)

        weights = np.zeros(patterns.shape[1])
        mse_ar = np.zeros(max_epochs)
        while epoch < max_epochs and mse > min_mse:
            # mse = 0
            y_pred = np.zeros(targets.shape)

            for i, pattern in enumerate(patterns):
                current_output = self.output(weights, pattern)
                # print(current_output)
                # print(targets[i])
                y_pred[i] = current_output
                weights = self.adjust_weights(
                    weights, targets[i], pattern, learning_rate, current_output)

                mse += pow((targets[i] - current_output), 2)
            mse = mse / 2.0
            mse_ar[epoch] = mse
            if plot:
                if(epoch % 10 == 0):
                    self.live_plot(axes, patterns, targets,
                                   y_pred, mse_ar, fig, epoch, d3)
            epoch += 1

            # plt.show()
        return weights

    def test(self, weights, patterns_test, targets_test):
        guesses = np.zeros(len(targets_test))
        for i in range(len(patterns_test)):
            guesses[i] = self.guess(weights, patterns_test[i])
        return guesses

    def plot_accuracy(self, targets, guesses):
        fig = plt.figure(2)
        fig.suptitle('Επιτυχείς/Ανεπιτυχείς κατανομές')

        ax = fig.add_subplot(1, 1, 1)
        # mple teleies: pragmatikoi stoxoi (y_test)
        plt.scatter(range(len(targets)), targets, marker='o', color='b')
        # kokkinoi kykloi: exwdos (predictions)
        plt.scatter(range(len(guesses)), guesses, marker='.', color='r')
        plt.xlabel("protypo")
        plt.ylabel("exodos (r) / stoxos (b)")
        y_true = np.reshape(targets, np.shape(guesses))
        accuracy = np.sum(y_true == guesses) / len(y_true)
        plt.show()
        print("Perceptron classification accuracy",
              accuracy*100, "%")

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
                                  i][j], c='tab:blue', marker='o', label='targets')
                axs[0, i].scatter(j, np.array(fold_targets)[
                                  i][j], c='tab:red', marker='.', label='guesses')
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
        plt.legend(['Guesses', 'Target'], bbox_to_anchor=(
            1, 1), bbox_transform=plt.gcf().transFigure)
        fig.suptitle(
            'Επιτυχείς/ανεπιτυχείς κατανομές (9-fold cross-validation)')
        # plt.show()

    def live_plot(self, axes, X, y, y_pred, mse, fig, epoch, d3):
        fig.suptitle('Epoch %d' % epoch)
        axes[0].clear()  # clear the line
        axes[1].clear()  # clear the line
        axes[2].clear()  # clear the line
        axes[3].clear()  # clear the line
        axes[0].scatter(X[:, 0], X[:, 1], marker='o', c=y)
        axes[1].scatter(X[:, 0], X[:, 1], marker='x', c=y_pred)
        if (d3):
            axes[0].scatter(X[:, 0], X[:, 2], marker='o', c=y)
            axes[1].scatter(X[:, 0], X[:, 2], marker='x', c=y_pred)

        axes[2].scatter(range(len(y_pred)), y_pred, marker='x', c=y_pred)
        axes[2].set_xlabel('x')
        axes[3].plot(range(len(mse)), mse, 'b')
        axes[3].set_xlabel('mse')

        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y ")
        axes[1].set_xlabel("x")
        plt.pause(0.001)
