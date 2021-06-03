from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.svm import SVR

mse = mae = float(0)


def regrevaluate(t, predict, criterion):
    if criterion == 'mse':
        value = np.mean((np.subtract(t, predict))**2)
    else:
        value = np.mean(np.abs(np.subtract(t, predict)))
    return value


# read from the file
boston_dataset = load_boston()

data = boston_dataset['data']

NumberOfAttributes = len(data[0, :])
NumberOfPatterns = len(data)


# initialize
x = data
t = boston_dataset['target']

#
gamma = [0.0001, 0.001, 0.01, 0.1]
C = [1, 10, 100, 1000]

# start_Of_folds
fig, subplt = plt.subplots(3, 3)

default = (int(input("run with default values?(1,yes/0,no): ") or 1) == 0)
kernel = input(
    'Δώσε kernel (linear, poly, rbf, sigmoid): ') or 'rbf'
gamma = float(
    input('Δώσε gamma 1, 0.1, 0.01, 0.001, 0.0001 : ') or 0.1)
C = int(input('Δώσε C 1, 10, 100, 1000, 10000, 100000: ') or 1)


n_folds = 9
for folds in range(0, n_folds):
    # liveplt.clear()
    xtrain, xtest, ttrain, ttest = train_test_split(x, t, test_size=0.25)

    numberOfTrain = len(xtrain)
    numberOfTest = len(xtest)

    xtrain = np.array(xtrain, dtype=float)
    xtest = np.array(xtest, dtype=float)
    if(default):
        model = SVR()
    else:
        model = SVR(C=C, kernel=kernel, gamma=gamma)

    model.fit(xtrain, ttrain)
    predict = model.predict(xtest)

    # mse += regrevaluate(ttest, predict, 'mse')
    # mae += regrevaluate(ttest, predict, 'msa')
    # print(xtest)
    # if (model.kernel == "linear"):

    #     xlim = liveplt.get_xlim()
    #     w = model.coef_[0]
    #     a = -w[0]/w[1]
    #     xx = np.linspace(xlim[0], xlim[1])
    #     yy = a*xx-(model.intercept_[0]/w[1])
    #     liveplt.scatter(xtest[:, 0], xtest[:, 1], marker='x', c=predict)

    #     liveplt.plot(xx, yy)
    # plots
    i = int((folds)/3)
    j = int((folds) % 3)
    subplt[i, j].plot(ttest, "ro")
    subplt[i, j].plot(predict, "b.")

print('Mean Squared Error for all folds is : %f \n', np.mean(mse))
print('Mean Absolute Error for all folds is : %f \n', np.mean(mae))
print('\n')
plt.show()
