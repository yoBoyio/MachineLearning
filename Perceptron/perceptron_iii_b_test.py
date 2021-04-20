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
input_file = "iris.csv"
df = pd.read_csv(input_file, header = None)
df.head()

# X: values, y: targets
# extract features
X = df.iloc[:, 0:3].values
# extract the label column
y = df.iloc[:, 4].values

# Setosa: 
y_setosa = np.where(y == 'Setosa', 1, 0)

# Versicolor:
y_versicolor = np.where(y == 'Versicolor', 1, 0)

# Virginica: 
y_virginica = np.where(y == 'Virginica', 1, 0)

sets =[y_setosa, y_versicolor, y_virginica]
predictions = []
y_test = []
i=0

for set in sets:

    # split data into train and test sets
    X_train, X_test, y_train, y_test_tmp = train_test_split(X, set, test_size=0.2, random_state=123)
    y_test.append(y_test_tmp)
    # print(X_train) # x,y values for training
    # print(X_test)  # x,y values for testing
    # print(y_train) # 0,1 targets for training
    # print(y_test[i])  # 0,1 targets for testing

    # create and train model
    p = Perceptron(learning_rate=0.01, n_iters=2000)
    p.fit(X_train, y_train)
    predictions.append(p.predict(X_test))

    # predictions: exodos, y_test: stoxos
    print("Perceptron classification accuracy", accuracy(y_test[i], predictions[i])*100, "%")
    i+=1

# plot

fig, (ax) = plt.subplots(1, 3, sharex=True, sharey=True)
for i in range(3):
    ax[i].scatter(range(len(y_test[i])), y_test[i], marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
    ax[i].scatter(range(len(predictions[i])), predictions[i], marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
    ax[i].set_xlabel("protypo")
    ax[i].set_ylabel("exodos (r) / stoxos (b)")

plt.show()