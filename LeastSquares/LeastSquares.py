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
        weights = np.ones(xtrain.shape) # allios epistrefei mono 0 otan kaleis to .dot 
        # print(weights)
        # print(np.transpose(nx))
        return weights.dot(nx)

    # Normalize value
    def normalize(self, val):
        #! edw bgazei poli mikra noumera, den 3erw ti paizei
        print(val)
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
            predictions.append(self.guess(xtest[index], weights[index])) #ayto ypotheto soy 3efige, alla den ebgaze poly nohma xwris to index sto weights
        return predictions
