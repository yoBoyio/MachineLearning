import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron


# calculate accuracy
def accuracy(y_true, y_pred):
    y_true = np.reshape(y_true, np.shape(y_pred))
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

def print_bitmap(X):
    fig, axes = plt.subplots(4,3)
    idx =0
    for i in range(4):
        for j in range(3):
            g = X[idx].reshape(11, 7)
            axes[i,j].imshow(g, cmap='Greys',  interpolation='nearest')
            idx+=1
    plt.show()

input_file = "bitmap_data.csv"
df = pd.read_csv(input_file, header = None)
df.head()

# X: values, y: targets
# extract features
X = df.iloc[:, 0:77].values
# extract the label column
y = df.iloc[:, 77].values

# Number 5: 
y_5 = np.where(y == 5, 1, 0)

# Number 6:
y_6 = np.where(y == 6, 1, 0)

# Number 8:
y_8 = np.where(y == 8, 1, 0)

# Number 9:
y_9 = np.where(y == 9, 1, 0)

sets =[y_5, y_6, y_8, y_9]
predictions = []
y_test = []
i=0

for set in sets:

    # split data into train and test sets
    X_train = np.concatenate((X[:8],X[11:19],X[22:30],X[33:41]))
    X_test = np.concatenate((X[8:11],X[19:22],X[30:33],X[41:]))

    y_train = np.concatenate((set[:8],set[11:19],set[22:30],set[33:41]))
    y_test_tmp = np.concatenate((set[8:11],set[19:22],set[30:33],set[41:]))

    if(i==0): #only print the first time
        print_bitmap(X_test)

    y_test.append(y_test_tmp)
    # print(X_train) # x,y values for training
    # print(X_test)  # x,y values for testing
    # print(y_train) # 0,1 targets for training
    # print(y_test[i])  # 0,1 targets for testing

    # create and train model
    p = Perceptron(learning_rate=0.01, n_iters=1000)
    p.fit(X_train, y_train)
    predictions.append(p.predict(X_test))

    # predictions: exodos, y_test: stoxos
    print("Perceptron classification accuracy for set %d" %(i+1), accuracy(y_test[i], predictions[i])*100, "%")
    i+=1

# plot

fig, (ax) = plt.subplots(1, 4, sharex=True, sharey=True)
for i in range(4):
    ax[i].scatter(range(len(y_test[i])), y_test[i], marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
    ax[i].scatter(range(len(predictions[i])), predictions[i], marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
    ax[i].set_xlabel("protypo")
    ax[i].set_ylabel("exodos (r) / stoxos (b)")

plt.show()