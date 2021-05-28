from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from functions import least_square_implementation


def ls():

    file = input("Δώσε input file(a,b,c,d): ")
    input_file = 'data_package_%s.csv' % file
    df = pd.read_csv(input_file, header=0)
    df = df._get_numeric_data()
    # targets
    targets_file = 'data_package_values_%s.csv' % file
    targets_df = pd.read_csv(targets_file, header=0)
    targets_df = targets_df._get_numeric_data()

    values = df.values
    targets = targets_df.values

    # plot_data(targets)
    # values = add_biases(values)
    patterns_train, patterns_test, targets_train, targets_test = train_test_split(
        values, targets, test_size=0.2, random_state=123)
    # plot_testing_data(targets_train, targets_test)
    plot = True
    d3 = False
    least_square_implementation(patterns_train, patterns_test,
                                targets_train, targets_test)


if __name__ == '__main__':
    ls()
