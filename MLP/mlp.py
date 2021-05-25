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

def live_plot(ax, X, y, y_pred, clf, mse_ar, epoch):
    # clear graphs
    ax[0][0].clear()
    ax[0][1].clear()
    ax[1][0].clear()
    ax[1][1].clear()
    # graph 1
    ax[0][0].scatter(X[:, 0], X[:, 1], marker='x', c=y)
    # graph 2
    plot_decision_regions(X=X, y=y_pred.flatten().astype(np.integer), clf=clf, legend=2,ax=ax[0][1])
    # graph 3
    for i in range(len(X)):
        if(y_pred[i]== 1):
            ax[1][0].scatter(i, y_pred[i], marker='x', c='y')
        else:
           ax[1][0].scatter(i, y_pred[i], marker='+', c='g')
    # graph 4
    ax[1][1].plot(range(len(mse_ar)), mse_ar, 'b')

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

def fit(clf,X_train, y_train, plot=False):
    if(plot):
        fig, axes = plt.subplots(2,2)
    mse_ar = np.zeros(max_iter)
    if (clf.solver =='sgd'): # partial_fit doesn't support sgd solver
        clf.fit(X_train,y_train)
        if(plot):
            live_plot(axes,X_train,y_train,clf.predict(X_train),clf,mse_ar,max_iter)
    else:
        for epoch in range(0,max_iter):
            clf.partial_fit(X_train, y_train.flatten(), classes=(np.unique(y_train)))
            mse = 1/y_train.shape[0] * pow(np.sum(np.subtract(y_train, clf.predict(X_train))), 2)
            mse_ar[epoch] = mse
            # 3. Monitor progress
            if(epoch%update_every==0): #update every X epochs
                print("Epoch %d" %epoch, "Accuracy %f" %clf.score(X_train, y_train))
                if(plot):
                    live_plot(axes,X_train,y_train,clf.predict(X_train),clf,mse_ar,epoch)
                    fig.suptitle('Epoch %d' % epoch)


while(True):
    plotting = int(input("0. Live Plot\n1. Plot Results\n2. Both\n") or 2) 
    file = input("Δώσε input file(a,b,c,d): ") or 'a'
    input_file = 'data_package_%s.csv' %file
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

    fit(clf, X_train, y_train, True if(plotting==0 or plotting==2)else False)

    if (plotting == 1 or plotting == 2):
        plot_results(clf,X_test, y_test)

    s = int(input('Δώσε 1 για να τρέξεις ξανά τον αλγόριθμο ή 0 για τερματισμό: ') or 0)
    if (s!=1): break
