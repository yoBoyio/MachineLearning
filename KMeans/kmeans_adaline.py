import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.core.defchararray import array
import pandas as pd
from sklearn import preprocessing
import numpy as np


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


def load_file():
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

    return X, y, is3d, isIris


def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (x-c)**2)


def plot(ax, X, y, classifications, centroids, is3d=False, isIris=False):
    ax[0].clear()
    ax[1].clear()
    col = 2 if (isIris) else 1  # emfanisi 3hs stilis gia iris
    colors = list(mcolors.TABLEAU_COLORS)
    if(is3d):
        ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], marker='x', c=y)
        for classification in classifications:
            color = colors[classification]
            for featureset in classifications[classification]:
                ax[1].scatter(featureset[0], featureset[1],
                              featureset[2], marker="x", c=color)
        for centroid in centroids:
            ax[1].scatter(centroids[centroid][0], centroids[centroid][1],
                          centroids[centroid][2], marker="*", color='r', s=150, alpha=0.6)
    else:
        ax[0].scatter(X[:, 0], X[:, col], marker='x', c=y)
        for classification in classifications:
            color = colors[classification]
            for featureset in classifications[classification]:
                ax[1].scatter(featureset[0], featureset[col],
                              marker="x", c=color)
        for centroid in centroids:
            ax[1].scatter(centroids[centroid][0], centroids[centroid][col],
                          marker="*", color='r', s=150, alpha=0.6)
    plt.pause(0.00001)


class KMeans:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y, live_plotting, update_plot, is3d, isIris):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = X[i]

        if (not is3d):
            fig, ax = plt.subplots(1, 2)
        if (is3d):
            fig = plt.figure(figsize=(8, 4))
            gs = fig.add_gridspec(1, 2)
            ax_3d_0 = fig.add_subplot(gs[0, 0], projection='3d')
            ax_3d_1 = fig.add_subplot(gs[0, 1], projection='3d')
            ax = [ax_3d_0, ax_3d_1]

        for epoch in range(1, self.max_iter+1):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X:
                distances = [np.linalg.norm(
                    featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(
                    self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            if(live_plotting and epoch % update_plot == 0 or epoch == self.max_iter or epoch == 1 or optimized):
                fig.suptitle(
                    ('Epoch %d' % epoch, ' optimized: %d' % optimized))
                plot(ax, X, y, self.classifications,
                     self.centroids, is3d, isIris)

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data-self.centroids[centroid])
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def main():
    plotting = int(input("0. Live Plot\n1. Only Results\n") or 0)
    live_plotting = plotting == 0

    X, y, is3d, isIris = load_file()

    clusters_num = int(input("Δώσε αριθμό κέντρων: ") or 2)
    max_iter = int(input('Δώσε αριθμό μέγιστων επαναλύψεων: ') or 300)
    update_plot = int(input("Update Plot ανά πόσες εποχές: ") or max_iter/10)
    tol = float(input('Δώσε tolerance: ') or 0.001)

    km = KMeans(k=clusters_num, max_iter=max_iter, tol=tol)
    km.fit(X, y, live_plotting, update_plot, is3d, isIris)

    dMax = max([np.abs(c1 - c2) for c1 in km.centroids for c2 in km.centroids])
    stds = np.repeat(dMax / np.sqrt(2*clusters_num), clusters_num)

    dist = []
    targets= []
    for centroid in km.centroids:
        tmp = []
        tmp_targets = []
        for classification in km.classifications:
            for featureset in km.classifications[classification]:
                r = rbf(featureset, km.centroids[centroid], stds[centroid])
                tmp.append(r)
                tmp_targets.append(centroid)
        tmp = np.array(tmp)
        tmp_targets = np.array(tmp_targets)
        dist.append(tmp)
        targets.append(tmp_targets)
    dist = np.array(dist)
    targets = np.array(targets)


    # plt.show()


if __name__ == '__main__':
    main()
