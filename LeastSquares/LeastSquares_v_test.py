from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from functions import least_square_implementation, add_biases


def print_bitmap(X):
    fig, axes = plt.subplots(4, 3)
    idx = 0
    for i in range(4):
        for j in range(3):
            g = X[idx].reshape(11, 7)
            axes[i, j].imshow(g, cmap='Greys',  interpolation='nearest')
            idx += 1
    # plt.show()


def ls_test_v_b():
    input_file = "bitmap_data.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:77].values
    X = add_biases(X)

    # extract the label column
    y = df.iloc[:, 77].values

    # Number 5:
    y_5 = np.where(y == 5, 1, 0)

    # Number 6:
    y_6 = np.where(y == 6, 1, 0)

    # Number 8:
    y_8 = np.where(y == 8, 1, 0)

    # Number 9:
    y_9 = np.where(y == 9, 1, 0)

    sets = [y_5, y_6, y_8, y_9]
    predictions = []
    y_test = []
    i = 0

    for set in sets:

        # split data into train and test sets
        X_train = np.concatenate((X[:8], X[11:19], X[22:30], X[33:41]))
        X_test = np.concatenate((X[8:11], X[19:22], X[30:33], X[41:]))

        y_train = np.concatenate((set[:8], set[11:19], set[22:30], set[33:41]))
        y_test_tmp = np.concatenate(
            (set[8:11], set[19:22], set[30:33], set[41:]))

        if(i == 0):  # only print the first time
            print_bitmap(X_test)

        y_test.append(y_test_tmp)
        # print(X_train)
        # print(y_train)

        # live plot
        plot = True
        # 3d plot
        d3 = False

        least_square_implementation(X_train, X_test,
                                    y_train, y_test_tmp)


if __name__ == '__main__':
    ls_test_v_b()
