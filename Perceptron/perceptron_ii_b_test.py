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
input_file = "data_package_ii_a.csv"
df = pd.read_csv(input_file, header = 0)
df = df._get_numeric_data()
# targets
targets_file = "data_package_values_ii_a.csv"
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
ax = plt.axes(projection='3d')

w,b = p.get_weights_bias()

ax.set_xlabel("X[0]")
ax.set_ylabel("X[1]")
ax.set_zlabel("X[2]")
# plot the samples
ax.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], marker='o', c=y_test, label='targets')
ax.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], marker='.', c=predictions, label='predictions') #kokkines teleies = lathos, kitrines = swsti ektimisi

w1 = w[0] #a
w2 = w[1] #b
w3 = w[2] #c

#construct hyperplane: ax + by + cz = d
a,b,c,d = w1,w2,w3,b

x_min = np.amin(X_test[:, 0])
x_max = np.amax(X_test[:, 0])
ax.set_xlim([x_min-0.2, x_max+0.2])

x = np.linspace(x_min, x_max, 100)

y_min = np.amin(X_test[:, 1])
y_max = np.amax(X_test[:, 1])
ax.set_ylim([y_min-0.2, y_max+0.2])

z_min = np.amin(X_test[:, 2])
z_max = np.amax(X_test[:, 2])
ax.set_zlim([z_min+0.2, z_max+0.2])

y = np.linspace(y_min, y_max, 100)

Xs,Ys = np.meshgrid(x,y)
Zs = ((d + a*Xs + b*Ys) / c)*(-1)

ax.plot_surface(Xs, Ys, Zs, alpha=0.45)

plt.legend()

plt.show()