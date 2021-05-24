import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron

def accuracy(y_true, y_pred):
    y_true = np.reshape(y_true, np.shape(y_pred))
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

# import and ready input file
input_file = "iris.csv"
df = pd.read_csv(input_file, header=None)
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

sets = [y_setosa, y_versicolor, y_virginica]
predictions = []
y_test = []
i = 0

for set in sets:

    # split data into train and test sets
    X_train, X_test, y_train, y_test_tmp = train_test_split(
        X, set, test_size=0.2, random_state=123)
    
    y_test.append(y_test_tmp)
    # print(X_train) # x,y values for training
    # print(X_test)  # x,y values for testing
    # print(y_train) # 0,1 targets for training
    # print(y_test[i])  # 0,1 targets for testing

    # create and train model
    p = Perceptron(learning_rate=0.01, n_iters=300)
    p.fit(X_train, y_train, plot=True)
    predictions.append(p.predict(X_test))

    # predictions: exodos, y_test: stoxos
    print("Perceptron classification accuracy",
          accuracy(y_test[i], predictions[i])*100, "%")
    i += 1

plt.show()
