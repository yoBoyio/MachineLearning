# Importing the libraries
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import datasets
import numpy as np

from SelfOrganizingMap import SOM
import pandas as pd


def plot(ax,  X, results, y, is3d, isIris):
    col = 2 if (isIris) else 1  # emfanisi 3hs stilis gia iris
    # ax[0].clear()
    # ax[1].clear()
    if(is3d):
        ax[0].scatter(X[:, 0], X[:, 2], marker='x', c=y)
        ax[1].scatter(X[:, 0], X[:, 2], marker='x', c=results)
        # for n in neurons:
        #     ax[1].scatter(n[0], n[1], n[2], marker='s',
        #                   color="r", s=100, alpha=0.7)
    else:
        ax[0].scatter([X[:, 0]], [X[:, col]], marker='o', c=y)
        ax[1].scatter([X[:, 0]], [X[:, col]], marker='o', c=results)
        # for n in neurons:
        #     ax[1].scatter(n[0], n[1], marker='s', color='r', s=100, alpha=0.7)
    plt.show()


def iris():
    # import and ready input file
    input_file = "iris.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:4].values
    # extract the label column
    y = df.iloc[:, 4].values

    y = np.where(y == 'Setosa', 0, y)
    y = np.where(y == 'Versicolor', 1, y)
    y = np.where(y == 'Virginica', 2, y)

    # standardise X
    standardized_X = preprocessing.scale(X)

    return(standardized_X, y)


if __name__ == '__main__':
    plotting = int(input("0. Live Plot\n1. Only Results\n") or 0)
    live_plotting = plotting == 0
    print('---------------------1.Load Data---------------------')
    d = 2
    file = input("Δώσε input file (a, b, c, d, ii_a, ii_b, iris): ") or 'a'
    is3d = file.__contains__("ii_")
    isIris = file.__contains__("iris")
    if (isIris):
        d = 4
        is3d = int(input("Δώσε 1 για εμφάνιση αποτελεσμάτων σε 3D: ") or 0) == 1
        X, y = iris()
    else:
        if(is3d):
            is3d = True
            d = 3
        input_file = 'data_package_%s.csv' % file
        df = pd.read_csv(input_file, header=0)
        df = df._get_numeric_data()
        X = df.values

        targets_file = 'data_package_values_%s.csv' % file
        targets_df = pd.read_csv(targets_file, header=0)
        targets_df = targets_df._get_numeric_data()
        y = targets_df.values

    m = int(input("Give vertical dimension "))
    n = int(input("Give horizontal dimension"))
    epoch = int(input("Give epochs "))

    som = SOM(m, n, d)

    # Fit it to the data
    som.fit(X, y, iris=isIris, plot_3d=is3d,
            plot_single=live_plotting, epochs=epoch)
    # Assign each datapoint to its predicted cluster
    predictions = som.predict(X)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    plot(ax, X, predictions, y, is3d, isIris)
