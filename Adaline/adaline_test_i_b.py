from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

from functions import plot_data, plot_testing_data, add_biases, adaline_implementation


def adaline_test_i_b():

    input_file = "data_package_b.csv"
    df = pd.read_csv(input_file, header=0)
    df = df._get_numeric_data()
    # targets
    targets_file = "data_package_values_b.csv"
    targets_df = pd.read_csv(targets_file, header=0)
    targets_df = targets_df._get_numeric_data()

    values = df.values
    targets = targets_df.values

    plot_data(targets)
    targets = add_biases(targets)
    patterns_train, patterns_test, targets_train, targets_test = train_test_split(
        values, targets, test_size=0.2, random_state=123)
    plot_testing_data(targets_train, targets_test)
    adaline_implementation(targets_train, targets_test,
                           patterns_train, patterns_test)


if __name__ == '__main__':
    adaline_test_i_b()
