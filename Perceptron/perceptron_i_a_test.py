import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron

# calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


file = input("Δώσε input file(a,b,c,d: ")
input_file = 'data_package_%s.csv' %file
df = pd.read_csv(input_file, header=0)
df = df._get_numeric_data()
# targets
targets_file = 'data_package_values_%s.csv' %file
targets_df = pd.read_csv(targets_file, header=0)
targets_df = targets_df._get_numeric_data()

# x: values, y: targets
X = df.values
y = targets_df.values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# create and train model
p = Perceptron(learning_rate=0.01, n_iters=100)
p.fit(X_train, y_train, True)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions), "%")
