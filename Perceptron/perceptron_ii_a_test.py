import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron


# calculate accuracy
def accuracy(y_true, y_pred):
    y_true = np.reshape(y_true, np.shape(y_pred))
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# import and ready input file
input_file = "data_package_ii_a.csv"
df = pd.read_csv(input_file, header=0)
df = df._get_numeric_data()
# targets
targets_file = "data_package_values_ii_a.csv"
targets_df = pd.read_csv(targets_file, header=0)
targets_df = targets_df._get_numeric_data()


# x: values, y: targets
X = df.values
y = targets_df.values
# display data
# for index in range(len(y)):
#     plt.scatter(index, 'class 1' if y[index] ==
#                 1 else 'class 0', marker='o', color='blue', label='correct')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)
# print(X_train) # x,y values for training
# print(X_test)  # x,y values for testing
# print(y_train) # 0,1 targets for training
# print(y_test)  # 0,1 targets for testing

# create and train model
p = Perceptron(learning_rate=0.01, n_iters=10)
p.fit(X_train, y_train, True)
predictions = p.predict(X_test)

# predictions: exodos, y_test: stoxos
print("Perceptron classification accuracy",
      accuracy(y_test, predictions)*100, "%")

# plot

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
# mple teleies: pragmatikoi stoxoi (y_test)
plt.scatter(range(len(y_test)), y_test, marker='o', color='b')
plt.scatter(range(len(predictions)), predictions, marker='.',
            color='r')  # kokkinoi kykloi: exwdos (predictions)
plt.xlabel("protypo")
plt.ylabel("exodos (r) / stoxos (b)")

plt.show()
