import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing


def normalization(M):
    return preprocessing.normalize(M)


def plot(ax, neurons, X, results, y, is3d, isIris):
    col = 2 if (isIris) else 1  # emfanisi 3hs stilis gia iris
    ax[0].clear()
    ax[1].clear()
    if(is3d):
        ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], marker='x', c=y)
        ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], marker='x', c=results)
        for n in neurons:
            ax[1].scatter(n[0], n[1], n[2], marker='s',
                          color="r", s=100, alpha=0.7)
    else:
        ax[0].scatter([X[:, 0]], [X[:, col]], marker='o', c=y)
        ax[1].scatter([X[:, 0]], [X[:, col]], marker='o', c=results)
        for n in neurons:
            ax[1].scatter(n[0], n[1], marker='s', color='r', s=100, alpha=0.7)
    plt.pause(0.00001)


def load_iris():
    # import and ready input file
    input_file = "iris.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:4].values
    # extract the label column
    y = df.iloc[:, 4].values

    y = np.where(y == 'Setosa', 0, y)
    y = np.where(y == 'Versicolor', 1, y)
    y = np.where(y == 'Virginica', 2, y)

    # standardise X
    standardized_X = preprocessing.scale(X)

    return(standardized_X, y)


class competitive_network(object):
    def __init__(self, x_dim, neurons_num, b):

        W = np.random.rand(neurons_num, x_dim)
        self.W = normalization(W)

        # βήμα
        self.b_init = b
        self.b = b

    # ds = |Ws-x| = min(Wk-x), k = 1,k
    def find_winner(self, x):
        distances = []
        for w in self.W:
            d = np.linalg.norm(w-x)
            distances.append(d)
        winnerIdx = np.argmin(distances)  # neuron closer to x
        return winnerIdx

    def train_winner(self, argmin, x):
        self.W[argmin] = self.W[argmin] + self.b * (x - self.W[argmin])

    def train(self, X, y, num_iter, update_plot, live_plotting, is3d, isIris=False):
        X = np.array(X)
        if (not is3d):
            fig, ax = plt.subplots(1, 2)
        if (is3d):
            fig = plt.figure(figsize=(8, 4))
            gs = fig.add_gridspec(1, 2)
            ax_3d_0 = fig.add_subplot(gs[0, 0], projection='3d')
            ax_3d_1 = fig.add_subplot(gs[0, 1], projection='3d')
            ax = [ax_3d_0, ax_3d_1]
        # show plot before any changes to the weights
        predict_results = self.prediction(X)
        fig.suptitle(('Epoch 0 Beta: %f' % self.b))
        plot(ax, self.get_neurons(), X, predict_results, y, is3d, isIris)
        for r in range(1, num_iter+1):
            self.b = self.b_init * (1-(r/num_iter*1.0))
            for i in range(X.shape[0]):
                winner = self.find_winner(X[i])
                self.train_winner(winner, X[i])
            if(live_plotting and r % update_plot == 0 or r == num_iter or r == 1):
                predict_results = self.prediction(X)
                fig.suptitle(('Epoch %d' % r, ' Beta: %f' % self.b))
                plot(ax, self.get_neurons(), X, predict_results, y, is3d, isIris)

    def prediction(self, X_test):
        sample_num = np.shape(X_test)[0]
        predict_results = []
        for i in range(sample_num):
            predict_result = self.find_winner(X_test[i])
            predict_results.append(predict_result)
        return predict_results

    def get_neurons(self):
        return self.W


def createCNN(X, y, live_plotting, is3d, isIris=False):
    print('------------------2.Parameters Seting----------------')
    neurons_num = int(input("Δώσε αριμθό νευρώνων: ") or 2)
    b = float(input("Δώσε αρχικό βήτα: ") or 0.999)
    num_iter = int(input("Δώσε αριμθό εποχών: ") or 1000)
    update_plot = int(input("Update Plot ανά πόσες εποχές: ") or num_iter/10)
    x_dim = np.shape(X)[1]

    print('-------------------3.Model Train---------------------')
    cnn = competitive_network(x_dim, neurons_num, b)
    cnn.train(X, y, num_iter, update_plot, live_plotting, is3d, isIris)

    print('-------------------4.Prediction----------------------')
    plt.show()


if __name__ == '__main__':
    plotting = int(input("0. Live Plot\n1. Only Results\n") or 0)
    live_plotting = plotting == 0
    print('---------------------1.Load Data---------------------')
    file = input("Δώσε input file (a, b, c, d, ii_a, ii_b, iris): ") or 'a'
    is3d = file.__contains__("ii_")
    isIris = file.__contains__("iris")
    if (isIris):
        is3d = int(input("Δώσε 1 για εμφάνιση αποτελεσμάτων σε 3D: ") or 0) == 1
        X, y = load_iris()
    else:
        input_file = 'data_package_%s.csv' % file
        df = pd.read_csv(input_file, header=0)
        df = df._get_numeric_data()
        X = df.values

        targets_file = 'data_package_values_%s.csv' % file
        targets_df = pd.read_csv(targets_file, header=0)
        targets_df = targets_df._get_numeric_data()
        y = targets_df.values

    createCNN(X, y, live_plotting, is3d, isIris)
