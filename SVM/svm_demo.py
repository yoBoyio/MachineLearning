
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


def create_SVC():

    if (int(input("run with default values?(1,yes/0,no): ") or 1) == 0):
        kernel = input('Δώσε kernel (linear, poly, rbf, sigmoid): ') or 'rbf'
        gamma = float(
            input('Δώσε gamma 1, 0.1, 0.01, 0.001, 0.0001 : ') or 0.1)
        C = int(input('Δώσε C 1, 10, 100, 1000, 10000, 100000: ') or 1)

        svc = SVC(C=C, kernel=kernel, gamma=gamma)
    else:
        svc = SVC()

    return svc


def iris(plotting_results, live_plotting):
    # import and ready input file
    input_file = "iris.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:4].values
    # extract the label column
    y = df.iloc[:, 4].values

    y_setosa = np.where(y == 'Setosa', 1, 0)
    y_versicolor = np.where(y == 'Versicolor', 1, 0)
    y_virginica = np.where(y == 'Virginica', 1, 0)
    sets = [y_setosa, y_versicolor, y_virginica]

    for set in sets:
        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, set, test_size=0.2, random_state=1)

        # create and train model
        svc = create_SVC()

        fit(svc, X_train, y_train, plot=live_plotting, isIris=True)

        if (plotting_results):
            plot_results(svc, X_test, y_test)


def housing(plotting_results, live_plotting):
    from sklearn.preprocessing import MinMaxScaler
    # input_file = 'housing.data'
    train_df = pd.read_csv(
        'https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/train.csv?alt=media&token=9c5d17c2-0589-43ea-b992-e7c2ad02d714', index_col='ID')
    train_df.head()
    test_df = pd.read_csv(
        'https://firebasestorage.googleapis.com/v0/b/bible-project-2365c.appspot.com/o/test.csv?alt=media&token=99688b27-9fdb-4ac3-93b8-fa0e0f4d7540', index_col='ID')
    test_df.head()

    predictors = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm',
                  'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']
    target = 'medv'

    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale both the training inputs and outputs
    scaled_train = scaler.fit_transform(train_df)
    print("Note: median values were scaled by multiplying by {:.10f} and adding {:.6f}".format(
        scaler.scale_[13], scaler.min_[13]))
    multiplied_by = scaler.scale_[13]
    added = scaler.min_[13]
    scaled_train_df = pd.DataFrame(
        scaled_train, columns=train_df.columns.values)

    X_train = scaled_train_df.drop(target, axis=1).values
    y_train = scaled_train_df[[target]].values

    # from sklearn.preprocessing import LabelBinarizer
    # y_dense = LabelBinarizer().fit_transform(y_train.astype(str))
    # print(y_dense)

    # y_train = y_dense

    svc = create_SVC()

    fit(svc, X_train, y_train, plot=live_plotting, isHousing=True)

    test_error_rate = svc.score(X=X_train, y=y_train)
    print("The mean squared error (MSE) for the test data set is: {}".format(
        test_error_rate))

    prediction = svc.predict(X_train)
    y_0 = prediction[0]
    print('Prediction with scaling - {}', format(y_0))
    y_0 -= added
    y_0 /= multiplied_by
    print("Housing Price Prediction  - ${}".format(y_0))

    Y_0 = y_train[0][0]
    print('Ground truth with scaling - {}'.format(Y_0))
    Y_0 -= added
    Y_0 /= multiplied_by

    print('Ground Truth Price - ${}'.format(Y_0))

    if (plotting_results):
        plot_results(svc, X_test, y_test)


def live_plot(ax, X, y, y_pred, svc,  isIris=False, isHousing=False):
    # clear graphs
    ax[0][0].clear()
    ax[0][1].clear()
    ax[1][0].clear()
    ax[1][1].clear()
    # graph 1
    ax[0][0].scatter(X[:, 0], X[:, 1], marker='x', c=y)
    # graph 2
    if (isIris):
        value = 1.5
        width = 0.75

    elif (isHousing):
        print("housing")
    else:
        if (svc.kernel == "linear"):

            xlim = ax[0][1].get_xlim()
            w = svc.coef_[0]
            a = -w[0]/w[1]
            xx = np.linspace(xlim[0], xlim[1])
            yy = a*xx-(svc.intercept_[0]/w[1])
            ax[0][1].scatter(X[:, 0], X[:, 1], marker='x', c=y_pred)

            ax[0][1].plot(xx, yy)

    # graph 3
    for i in range(len(X)):
        if(y_pred[i] == 1):
            ax[1][0].scatter(i, y_pred[i], marker='x', c='y')
        else:
            ax[1][0].scatter(i, y_pred[i], marker='+', c='g')
    # graph 4

    plt.pause(0.0001)


def live_plot_3d(ax, X, y, y_pred, svc):
    ax[0].clear()
    ax[1].clear()
    ax[2].clear()

    ax[0].scatter(X[:, 0], X[:, 1], X[:, 2], marker='x', c=y)
    ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], marker='x', c=y_pred)
    ax[2].scatter(range(len(y_pred)), y_pred, marker='x', c=y)

    if (svc.kernel == "linear"):
        w = svc.coef_[0]
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        b = svc.intercept_[0]

        def z(x, y): return (-b-w1*x-w2*y) / w3

        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        ax[1].set_xlim([x_min-0.2, x_max+0.2])
        x = np.linspace(x_min, x_max, 100)
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])
        ax[1].set_ylim([y_min-0.2, y_max+0.2])
        z_min = np.amin(X[:, 2])
        z_max = np.amax(X[:, 2])
        y = np.linspace(y_min, y_max, 100)
        ax[1].set_zlim([z_min+0.2, z_max+0.2])

        x, y = np.meshgrid(x, y)
        ax[1].plot_surface(x, y, z(x, y), alpha=0.3)

    plt.pause(0.0001)


def fit(svc, X_train, y_train, plot=False, plot_3d=False, isIris=False, isHousing=False):
    if(plot):
        fig, axes = plt.subplots(2, 2)
    if(plot_3d):
        fig_3d = plt.figure(figsize=(12, 4))
        gs = fig_3d.add_gridspec(1, 3)
        ax_3d_0 = fig_3d.add_subplot(gs[0, 0], projection='3d')
        ax_3d_1 = fig_3d.add_subplot(gs[0, 1], projection='3d')
        ax_3d_2 = fig_3d.add_subplot(gs[0, 2])
        ax3d = [ax_3d_0, ax_3d_1, ax_3d_2]

    svc.fit(X_train, y_train)
    if(plot):
        live_plot(axes, X_train, y_train, svc.predict(X_train),
                  svc, isIris=isIris, isHousing=isHousing)
    if(plot_3d):
        live_plot_3d(ax3d, X_train, y_train, svc.predict(X_train), svc)


def plot_results(svc, X_test, y_test):
    predictions = svc.predict(X_test)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # mple teleies: pragmatikoi stoxoi (y_test)
    ax.scatter(range(len(y_test)), y_test,
               marker='o', color='b', label="targets")
    # kokkinoi kykloi: exwdos (predictions)
    ax.scatter(range(len(predictions)), predictions,
               marker='.', color='r', label="predictions")
    ax.set_xlabel("πρότυπο")
    ax.set_ylabel("έξοδος / στόχος")
    ax.legend()
    plt.show()


def plot_results_3d(svc, X_test, y_test):
    plot_results(svc, X_test, y_test)


# main
while(True):
    plotting = int(input("0. Live Plot\n1. Plot Results\n2. Both\n") or 2)
    live_plotting = plotting == 0 or plotting == 2
    plotting_results = plotting == 1 or plotting == 2
    live_plotting_3d, plotting_results_3d = False, False

    file = input(
        "Δώσε input file (a, b, c, d, ii_a, ii_b, iris, housing): ") or 'a'
    if (file == "iris"):
        iris(plotting_results, live_plotting)
    elif(file == "housing"):
        housing(live_plotting=live_plotting, plotting_results=plotting_results)

    else:
        input_file = 'data_package_%s.csv' % file
        if (file.__contains__("ii_")):
            live_plotting_3d, plotting_results_3d = live_plotting, plotting_results
            live_plotting, plotting_results = False, False
        df = pd.read_csv(input_file, header=0)
        df = df._get_numeric_data()
        # targets
        targets_file = 'data_package_values_%s.csv' % file
        targets_df = pd.read_csv(targets_file, header=0)
        targets_df = targets_df._get_numeric_data()

        # X: values, y: targets
        X = df.values
        y = targets_df.values

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1)

        svc = create_SVC()

        fit(svc, X_train, y_train, plot=live_plotting, plot_3d=live_plotting_3d)

        if (plotting_results):
            plot_results(svc, X_test, y_test)
        if (plotting_results_3d):
            plot_results_3d(svc, X_test, y_test)

    s = int(input('Δώσε 1 για να τρέξεις ξανά τον αλγόριθμο ή 0 για τερματισμό: ') or 0)
    if (s != 1):
        break
