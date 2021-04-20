import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron


# calculate accuracy
def accuracy(y_true, y_pred):
    y_true = np.reshape(y_true,np.shape(y_pred))
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# import and ready input file
input_file = "data_package_ii_b.csv"
df = pd.read_csv(input_file, header = 0)
df = df._get_numeric_data()
# targets
targets_file = "data_package_values_ii_b.csv"
targets_df = pd.read_csv(targets_file, header = 0)
targets_df = targets_df._get_numeric_data()


# x: values, y: targets
X = df.values
y = targets_df.values

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# print(X_train) # x,y values for training
# print(X_test)  # x,y values for testing
# print(y_train) # 0,1 targets for training
# print(y_test)  # 0,1 targets for testing

# create and train model
p = Perceptron(learning_rate=0.01, n_iters=1000)
p.fit(X_train, y_train)
predictions = p.predict(X_test)

# predictions: exodos, y_test: stoxos
print("Perceptron classification accuracy", accuracy(y_test, predictions)*100, "%")

# plot

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(range(len(y_test)), y_test, marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
plt.scatter(range(len(predictions)), predictions, marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
plt.xlabel("protypo")
plt.ylabel("exodos (r) / stoxos (b)")

# TODO diaxoristiki grammi?

# x0_1 = np.amin(X_test[:, 0])
# x0_2 = np.amax(X_test[:, 0])
# x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
# x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
# ax.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

plt.show()