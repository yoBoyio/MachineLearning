from math import e
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_decision_regions


def create_MLP():
    global clf
    global max_iter
    global update_every
    update_every = 1
    max_iter = 200
    if (int(input("run with default values?(1,yes/0,no): ") or 1) == 0):
        solver = input('Δώσε solver (adam, sgd, lbfgs): ') or 'adam'
        hidden_layer_size = (int(input("Δώσε αριθμό κρυφών νευρώνων: ") or 100),)
        activation = 'logistic' if (int(input('Δώσε 0 για ‘tanh’ (δηλ. σιγμοειδή -1/1) ή 1 για ‘logistic’ (δηλ. σιγμοειδή 0/1): ') or 0) == 1) else 'tanh'
        max_iter = int(input('Δώσε max_iter: ') or 100)
        update_every = int(input('Ανανέωση Live Plot ανά πόσες εποχές? ') or 1) 
        learning_rate = input("Δώσε learning_rate (constant, invscaling, adaptive): ") or 'constant'
        learning_rate_init = float(input('Δώσε learning_rate_init (πχ 0.001): ') or 0.001)
        momentum = float(input('Δώσε momentum (πχ 0.9). Χρησιμοποιείται μόνο όταν solver="sgd": ') or 0)

        clf = MLPClassifier(
            random_state=1,
            solver=solver,
            max_iter=max_iter,
            hidden_layer_sizes=hidden_layer_size,
            activation=activation,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            momentum=momentum
            )
    else:
        clf = MLPClassifier()

    if(update_every <= 0): update_every = 1

    return clf

def live_plot(ax, X, y, y_pred, clf, mse_ar, isIris=False):
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
        plot_decision_regions(X, y=y_pred.flatten().astype(np.integer), clf=clf,
            filler_feature_values={2: value},
            filler_feature_ranges={2: width},
            legend=2, ax=ax[0][1])
    else:
        plot_decision_regions(X, y=y_pred.flatten().astype(np.integer), clf=clf, legend=2, ax=ax[0][1])

    # graph 3
    for i in range(len(X)):
        if(y_pred[i]== 1):
            ax[1][0].scatter(i, y_pred[i], marker='x', c='y')
        else:
           ax[1][0].scatter(i, y_pred[i], marker='+', c='g')
    # graph 4
    ax[1][1].plot(range(len(mse_ar)), mse_ar, 'b')

    plt.pause(0.0001)

def live_plot_3d(ax, X, y, y_pred, clf):
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].scatter(X[:, 0], X[:, 1],X[:, 2], marker='x', c=y)        
        ax[1].scatter(X[:, 0], X[:, 1],X[:, 2], marker='x', c=y_pred)
        ax[2].scatter(range(len(y_pred)), y_pred,marker='x', c=y)

        epoch = clf.n_iter_-1
        w = clf.coefs_[0]
        w1 = w[0]
        w2 = w[1]
        w3 = w[2]
        b = clf.intercepts_[0]


        z = lambda x,y: (-b-w1*x-w2*y) / w3

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

        x,y = np.meshgrid(x,y)
        ax[1].plot_surface(x, y, z(x,y), alpha=0.3)

        plt.pause(0.0001)

def plot_results(clf, X_test, y_test):
    predictions = clf.predict(X_test)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # mple teleies: pragmatikoi stoxoi (y_test)
    ax.scatter(range(len(y_test)), y_test, marker='o', color='b', label="targets")
    # kokkinoi kykloi: exwdos (predictions)
    ax.scatter(range(len(predictions)), predictions, marker='.', color='r', label="predictions")
    ax.set_xlabel("πρότυπο")
    ax.set_ylabel("έξοδος / στόχος")
    ax.legend()
    plt.show()

def plot_results_3d(clf, X_test, y_test):
   plot_results(clf, X_test, y_test)

def fit(clf,X_train, y_train, plot=False, plot_3d=False, isIris=True):
    if(plot):
        fig, axes = plt.subplots(2,2)
    if(plot_3d):
        fig_3d = plt.figure(figsize=(12,4))
        gs = fig_3d.add_gridspec(1, 3)
        ax_3d_0 = fig_3d.add_subplot(gs[0,0], projection='3d')
        ax_3d_1= fig_3d.add_subplot(gs[0,1], projection='3d')
        ax_3d_2= fig_3d.add_subplot(gs[0,2])
        ax3d=[ax_3d_0,ax_3d_1,ax_3d_2]
    mse_ar = np.zeros(max_iter)
    if (clf.solver =='sgd'): # partial_fit doesn't support sgd solver
        clf.fit(X_train,y_train)
        if(plot):
            live_plot(axes,X_train,y_train,clf.predict(X_train),clf,mse_ar, isIris=isIris)
        if(plot_3d):
            live_plot_3d(ax3d, X_train, y_train, clf.predict(X_train), clf)
    else:
        for epoch in range(0,max_iter):
            clf.partial_fit(X_train, y_train.flatten(), classes=(np.unique(y_train)))
            mse = 1/y_train.shape[0] * pow(np.sum(np.subtract(y_train, clf.predict(X_train))), 2)
            mse_ar[epoch] = mse
            # 3. Monitor progress
            if(epoch%update_every==0): #update every X epochs
                print("Epoch %d" %epoch, "Accuracy %f" %clf.score(X_train, y_train))
                if(plot):
                    live_plot(axes,X_train,y_train,clf.predict(X_train),clf,mse_ar, isIris=isIris)
                    fig.suptitle('Epoch %d' % epoch)
                if(plot_3d):
                    live_plot_3d(ax3d, X_train, y_train, clf.predict(X_train), clf)

def iris(plotting_results, live_plotting):
    # import and ready input file
    input_file = "iris.csv"
    df = pd.read_csv(input_file, header=None)
    df.head()

    # X: values, y: targets
    # extract features
    X = df.iloc[:, 0:3].values
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
        clf = create_MLP()

        fit(clf, X_train, y_train, plot=live_plotting, isIris= True)

        if (plotting_results):
            plot_results(clf,X_test, y_test)

while(True):
    plotting = int(input("0. Live Plot\n1. Plot Results\n2. Both\n") or 2)
    live_plotting = plotting==0 or plotting==2
    plotting_results = plotting == 1 or plotting == 2
    live_plotting_3d, plotting_results_3d = False, False
    
    file = input("Δώσε input file(a, b, c, d, ii_a, ii_b, iris): ") or 'a'
    if (file == "iris"):
        iris(plotting_results, live_plotting)
    else:
        input_file = 'data_package_%s.csv' %file
        if (file.__contains__("ii_")):
            live_plotting_3d, plotting_results_3d = live_plotting, plotting_results
            live_plotting, plotting_results = False, False
        df = pd.read_csv(input_file, header=0)
        df = df._get_numeric_data()
        # targets
        targets_file = 'data_package_values_%s.csv' %file
        targets_df = pd.read_csv(targets_file, header=0)
        targets_df = targets_df._get_numeric_data()

        # X: values, y: targets
        X = df.values
        y = targets_df.values

        # split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        clf = create_MLP()

        fit(clf, X_train, y_train, plot=live_plotting, plot_3d=live_plotting_3d)

        if (plotting_results):
            plot_results(clf,X_test, y_test)
        if (plotting_results_3d):
            plot_results_3d(clf,X_test, y_test)

    s = int(input('Δώσε 1 για να τρέξεις ξανά τον αλγόριθμο ή 0 για τερματισμό: ') or 0)
    if (s!=1): break