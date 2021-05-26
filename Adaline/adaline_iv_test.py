from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

from functions import plot_data, plot_testing_data, add_biases, adaline_implementation
from sklearn.preprocessing import MinMaxScaler


def adaline_test_i_b():

    # import and ready house prediction data
    train_df = pd.read_csv(
        'https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/train.csv?alt=media&token=9c5d17c2-0589-43ea-b992-e7c2ad02d714', index_col='ID')
    train_df.head()

    test_df = pd.read_csv(
        'https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/test.csv?alt=media&token=99688b27-9fdb-4ac3-93b8-fa0e0f4d7540', index_col='ID')
    test_df.head()

    predictors = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                  'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    target = 'dis'

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale both the training inputs and outputs
    scaled_train = scaler.fit_transform(train_df)

    # scaled_test = scaler.transform(test_df)
    # Print out the adjustment that the scaler applied to the total_earnings column of data
    # print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(
    # scaler.scale_[13], scaler.min_[13]))

    multiplied_by = scaler.scale_[13]
    ded = scaler.min_[13]

    scaled_train_df = pd.DataFrame(
        scaled_train, columns=train_df.columns.values)

    # x: values, y: targets
    X = scaled_train_df.drop(target, axis=1).values
    Y = scaled_train_df[[target]].values
    X_train, X_test, y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=123)
    # live plot
    plot = True
    # 3d plot
    d3 = True
    adaline_implementation(y_train, Y_test,
                           X_train, X_test, plot, d3)


if __name__ == '__main__':
    adaline_test_i_b()
