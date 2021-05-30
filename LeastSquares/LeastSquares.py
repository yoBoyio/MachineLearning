import numpy as np
import matplotlib.pyplot as plt


class LeastSquares():

    def transmute_targets(self, targets_train, targets_test):
        for index, target in enumerate(targets_train):
            if target == 0:
                targets_train[index] = -1
        for index, target in enumerate(targets_test):
            if target == 0:
                targets_test[index] = -1
        return targets_train, targets_test

    def get_weights(self, xtrain, ttrain):
        xt = []
        for item in xtrain:
            temp = []
            for nump in item:
                temp.append(nump)
            xt.append(temp)
        x = np.linalg.pinv(xt)
        t = np.asarray(ttrain)
        nx = np.asarray(x)
        return t.dot(np.transpose(nx))

    def test(self, weights, patterns_test, targets_test):
        guesses = np.zeros(len(targets_test))
        for i in range(len(patterns_test)):
            guesses[i] = self.guess(weights, patterns_test[i])
        return guesses

    def guess(self, weights, pattern):
        weighted_sum = 0
        for i in range(len(pattern)):
            weighted_sum += weights[i] * pattern[i]
        return self.sign_function(weighted_sum)

    def sign_function(self, weighted_sum):
        if weighted_sum < 0:
            return -1
        else:
            return 1

    def get_accuracy(self, guesses, targets):
        correct_answers = 0
        for i in range(len(guesses)):
            if guesses[i] == targets[i]:
                correct_answers += 1
        return int((correct_answers / len(targets)) * 100)
