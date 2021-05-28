import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

from perceptron import Perceptron


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


def plot_results_3d(p,X_test,y_test,predictions):
    fig = plt.figure()
    fig.suptitle("Results")
    ax = plt.axes(projection='3d')

    w,b = p.get_weights_bias()

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_zlabel("X[2]")
    # plot the samples
    ax.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], marker='o', c=y_test, label='targets')
    ax.scatter(X_test[:, 0], X_test[:, 1],X_test[:, 2], marker='.', c=predictions, label='predictions') #kokkines teleies = lathos, kitrines = swsti ektimisi

    w1 = w[0] #a
    w2 = w[1] #b
    w3 = w[2] #c

    #construct hyperplane: ax + by + cz = d
    a,b,c,d = w1,w2,w3,b

    x_min = np.amin(X_test[:, 0])
    x_max = np.amax(X_test[:, 0])
    ax.set_xlim([x_min-0.2, x_max+0.2])

    x = np.linspace(x_min, x_max, 100)

    y_min = np.amin(X_test[:, 1])
    y_max = np.amax(X_test[:, 1])
    ax.set_ylim([y_min-0.2, y_max+0.2])

    z_min = np.amin(X_test[:, 2])
    z_max = np.amax(X_test[:, 2])
    ax.set_zlim([z_min+0.2, z_max+0.2])

    y = np.linspace(y_min, y_max, 100)

    Xs,Ys = np.meshgrid(x,y)
    Zs = ((d + a*Xs + b*Ys) / c)*(-1)

    ax.plot_surface(Xs, Ys, Zs, alpha=0.45)

    plt.legend()

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

        # create and train model
        p = Perceptron(learning_rate=0.01, n_iters=300)
        p.fit(X_train, y_train, plot=live_plotting)
        predictions.append(p.predict(X_test))

        # predictions: exodos, y_test: stoxos
        print("Perceptron classification accuracy",
            accuracy(y_test[i], predictions[i])*100, "%")
        i += 1
    
    if(plotting_results):
        fig, (ax) = plt.subplots(1, 3, sharex=True, sharey=True)
        fig.suptitle("Results")
        for i in range(3):
            ax[i].scatter(range(len(y_test[i])), y_test[i], marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
            ax[i].scatter(range(len(predictions[i])), predictions[i], marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
            ax[i].set_xlabel("protypo")
            ax[i].set_ylabel("exodos (r) / stoxos (b)")

def bitmap(plotting_results, live_plotting):
    def print_bitmap(X):
        fig, axes = plt.subplots(4,3)
        fig.suptitle("Results")
        idx =0
        for i in range(4):
            for j in range(3):
                g = X[idx].reshape(11, 7)
                axes[i,j].imshow(g, cmap='Greys',  interpolation='nearest')
                idx+=1

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
        # create and train model
        p = Perceptron(learning_rate=0.01, n_iters=200)
        p.fit(X_train, y_train, plot_single = live_plotting)
        predictions.append(p.predict(X_test))
        # predictions: exodos, y_test: stoxos
        print("Perceptron classification accuracy for set %d" %(i+1), accuracy(y_test[i], predictions[i])*100, "%")
        i+=1
    
    if(plotting_results):
        fig, (ax) = plt.subplots(1, 4, sharex=True, sharey=True)
        for i in range(4):
            ax[i].scatter(range(len(y_test[i])), y_test[i], marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
            ax[i].scatter(range(len(predictions[i])), predictions[i], marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
            ax[i].set_xlabel("protypo")
            ax[i].set_ylabel("exodos (r) / stoxos (b)")

def main():
    plotting = int(input("0. Live Plot\n1. Plot Results\n2. Both\n") or 2)
    live_plotting = plotting == 0 or plotting == 2
    plotting_results = plotting == 1 or plotting == 2
    live_plotting_3d, plotting_results_3d = False, False

    file = input("Δώσε input file (a, b, c, d, ii_a, ii_b, iris, bitmap): ") or 'a'
    if (file == "iris"):
            iris(plotting_results, live_plotting)
    elif(file == "bitmap"):
        bitmap(plotting_results, live_plotting)
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

            # x: values, y: targets
            X = df.values
            y = targets_df.values

            # split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=123)

            # create and train model
            p = Perceptron(learning_rate=0.01, n_iters=100)
            p.fit(X_train, y_train, plot=live_plotting, plot_3d=live_plotting_3d)
            predictions = p.predict(X_test)

            if(plotting_results):
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                # mple teleies: pragmatikoi stoxoi (y_test)
                plt.scatter(range(len(y_test)), y_test, marker='o', color='b')
                plt.scatter(range(len(predictions)), predictions, marker='.',
                            color='r')  # kokkinoi kykloi: exwdos (predictions)
                plt.xlabel("protypo")
                plt.ylabel("exodos (r) / stoxos (b)")
            if(plotting_results_3d):
                plot_results_3d(p,X_test,y_test,predictions)
            print("Perceptron classification accuracy", accuracy(y_test, predictions), "%")
    plt.show()

if( __name__ == "__main__"):
    main()