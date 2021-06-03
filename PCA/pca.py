import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def accuracy(y_true, y_pred, isTest):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    if(isTest):
        print('Test accuracy: %f' % accuracy)
    else:
        print('Train accuracy: %f' % accuracy)
    return accuracy


data = np.load('pca_data.npz')
X = data['x']
y = data['t']

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# accuracy(y_train, gnb.predict(X_train),isTest=False)
# accuracy(y_test, gnb.predict(X_test),isTest=True)

acc_train = []
acc_test = []

fig, ax = plt.subplots(1, 1)
fig.suptitle("Results")
ax.set_xlabel("num components")
ax.set_ylabel("accuracy")

num_components_ar = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200]
i=0
for num_components in num_components_ar:
    ax.clear()
    pca = PCA(n_components=num_components)
    x_pca = pca.fit_transform(X)
    # split data into train and test sets
    X_pca_train, X_pca_test, y_train, y_test = train_test_split(
        x_pca, y, test_size=0.2, random_state=1)
    gnb.fit(X_pca_train, y_train)

    print('\nNumber of components: %d' % num_components)
    acc_train.append(accuracy(y_train, gnb.predict(X_pca_train), isTest=False))
    acc_test.append(accuracy(y_test, gnb.predict(X_pca_test), isTest=True))

    ax.plot(num_components_ar[:i+1], acc_train[:i+1], 'bo')
    ax.plot(num_components_ar[:i+1], acc_train[:i+1], c='b', label='train')
    ax.plot(num_components_ar[:i+1], acc_test[:i+1], 'ro')
    ax.plot(num_components_ar[:i+1], acc_test[:i+1], c='r', label='test')
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Accuracy")
    i= i + 1
    plt.legend()
    plt.pause(0.0001)
    
plt.show()
