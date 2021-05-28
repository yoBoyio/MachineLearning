import numpy as np


class LeastSquares:

    # Get correct weight
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
        # t = np.transpose(t)
        weights = np.zeros(xtrain.shape)
        print(weights)
        print(np.transpose(nx))
        return weights.dot(np.transpose(nx))

    # Normalize value
    def normalize(self, val):
        if val < 0:
            return -1.0
        else:
            return 1.0

    # Find guess and return normalized value
    def guess(self, xtrain, weights):
        weighted_sum = 0.0
        for i in range(len(xtrain)):
            weighted_sum += xtrain[i]*weights[i]
        return self.normalize(weighted_sum)

    # Get predictions
    def get_predictions(self, xtest, weights):
        predictions = []
        for index in range(len(xtest)):
            predictions.append(self.guess(xtest[index], weights))
        return predictions
