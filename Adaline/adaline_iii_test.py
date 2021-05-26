from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from functions import plot_data, plot_testing_data, add_biases, adaline_implementation


def adaline_test_i_b():

    # import and ready input file
    input_file = "iris.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:3].values
    # extract the label column
    y = df.iloc[:, 4].values
    # normalize the data attributes
    X = preprocessing.normalize(X)
    # TODO
    # add biases
    X = add_biases(X)

    # Setosa:
    y_setosa = np.where(y == 'Setosa', 1, 0)

    # Versicolor:
    y_versicolor = np.where(y == 'Versicolor', 1, 0)

    # Virginica:
    y_virginica = np.where(y == 'Virginica', 1, 0)

    sets = [y_setosa, y_versicolor, y_virginica]
    predictions = []
    y_test = []
    i = 0

    for set in sets:

        # split data into train and test sets
        X_train, X_test, y_train, y_test_tmp = train_test_split(
            X, set, test_size=0.2, random_state=123)

        y_test.append(y_test_tmp)

        # live plot
        plot = True
        # 3d plot
        d3 = True
        adaline_implementation(y_train, y_test_tmp,
                               X_train, X_test, plot, d3)


if __name__ == '__main__':
    adaline_test_i_b()
