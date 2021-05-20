import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron


# calculate accuracy
def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# import and ready input file
input_file = "data_package_a.csv"
df = pd.read_csv(input_file, header=0)
df = df._get_numeric_data()
# targets
targets_file = "data_package_values_a.csv"
targets_df = pd.read_csv(targets_file, header=0)
targets_df = targets_df._get_numeric_data()

# x: values, y: targets
X = df.values
y = targets_df.values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123)

# create and train model

p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

print("Perceptron classification accuracy", accuracy(y_test, predictions), "%")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', c=y_train)

# plt.scatter(range(len(predictions)), predictions, marker='.',
#             color='r')  # kokkinoi kykloi: exwdos (predictions)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])
ax.set_ylim([ymin-3, ymax+3])

plt.xlabel("protypo")
plt.ylabel("exodos ")
plt.show()
