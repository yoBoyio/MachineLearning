import numpy as np
from sklearn.model_selection import train_test_split
from functions import adaline_implementation, add_biases
from kmeans import KMeans, load_file


def rbf(x, c, s):
    return np.exp(-1 / (2 * s**2) * (np.linalg.norm(x-c))**2)

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
    std = dMax / np.sqrt(2*clusters_num)

    patterns = []
    targets = []
    for classification in km.classifications:
        for featureset in km.classifications[classification]:
            tmp = []
            for centroid in km.centroids:
                r = rbf(featureset, km.centroids[centroid], std)
                tmp.append(r)
            tmp = np.array(tmp)
            patterns.append(tmp)
            targets.append(classification)
    patterns = np.array(patterns)
    targets = np.array(targets)

    print(patterns)
    print(targets)

    patterns = add_biases(patterns)
    patterns_train, patterns_test, targets_train, targets_test = train_test_split(
        patterns, targets, test_size=0.2, random_state=123)

    adaline_implementation(targets_train, targets_test, patterns_train,
                           patterns_test, plot=live_plotting, d3=False)


if __name__ == '__main__':
    main()
