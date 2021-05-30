import numpy as np
# from sources.generator_distributions import mix_distributions, generate_normal_distributions, generate_expon_distributions, get_y_train
# from sources import config
from math import sqrt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import preprocessing


def sum_squares(arr):
    return sum([x ** 2 for x in arr])

# function normalization for vectors X


def normalization(arr_x, arr_y):
    sum_y_squares = sum_squares(arr_y)
    result = []

    for row_x in arr_x:
        middle_result = []
        sum_x_squares = sum_squares(row_x)
        for x in row_x:
            middle_result.append(
                round(x / sqrt(sum_x_squares + sum_y_squares), 2))

        result.append(middle_result)
    return result


class Counter_Propagation:
    def __init__(self, X_values, y_values, kohonen_neurons=2, grossberg_neurons=1, len_x_vector=4):
        self.X_values = X_values
        self.y_values = y_values

        self.kohonen_weights = self.generate_weights(
            kohonen_neurons, len_x_vector)
        self.grossberg_weights = self.generate_weights(
            grossberg_neurons, len_x_vector)

    # generate weights for each part of network
    def generate_weights(self, num_neurons=1, length=4):
        result = np.asarray(np.random.rand(num_neurons, length))
        if len(result) == 1:
            return result[0]
        return result

    # for shorter Evklide's way
    def calculate_evklid_way(self, w_vector, x_vector):
        return sum([((w-x) ** 2) for w, x in zip(w_vector, x_vector)])

    # calculate net for Grossberg lay
    def sum_activation(self, k_vector, w_vector):
        return sum([k*w for k, w in zip(k_vector, w_vector)])

    # update vector kohonen weights
    def update_kohonen_weights(self, x_vector, w_vector, learning_rate=0.7):
        weights = []

        for x, w in zip(x_vector, w_vector):
            w_new = w + learning_rate * (x - w)
            weights.append(w_new)

        return np.asarray(weights)

    # update weights for grossberg lay
    def update_grossberg_weights(self, y_value, w_value, learning_rate=0.1, k=1):
        w_new = w_value + learning_rate * (y_value - w_value) * k
        return w_new

    # counter of training success
    def good_count(self, y_value, out_network):
        if y_value == out_network:
            return 1
        return 0

    # training counter propagation neural network
    def fit(self, lr_kohonen=0.7, lr_grossberg=0.1, epochs=10, plot_3d=False, iris=False):

        if plot_3d:
            fig_3d = plt.figure(figsize=(8, 4))
            gs = fig_3d.add_gridspec(1, 2)
            ax_3d_0 = fig_3d.add_subplot(gs[0, 0], projection='3d')
            ax_3d_1 = fig_3d.add_subplot(gs[0, 1], projection='3d')
            ax3d = [ax_3d_0, ax_3d_1]
        else:
            fig, axs = plt.subplots(1, 2)

        for epoch in range(epochs):
            y_pred = np.zeros(self.y_values.shape)

            if epoch % 5 == 0 and lr_kohonen > 0.1 and lr_grossberg > 0.01:
                lr_kohonen -= 0.05
                lr_grossberg -= 0.005

            good_counter = 0
            idx = 0

            for x_vector, y_value in zip(self.X_values, self.y_values):
                kohonen_neurons = []
                for w_iter in range(len(self.kohonen_weights)):
                    kohonen_neurons.append(self.calculate_evklid_way(
                        x_vector, self.kohonen_weights[w_iter]))
                neuron_min = min(kohonen_neurons)
                index = kohonen_neurons.index(neuron_min)

                for i in range(len(kohonen_neurons)):
                    if i == index:
                        kohonen_neurons[i] = 1
                    else:
                        kohonen_neurons[i] = 0

                self.kohonen_weights[index] = self.update_kohonen_weights(
                    x_vector, self.kohonen_weights[index], learning_rate=lr_kohonen)

                # grossberg neurons
                self.grossberg_weights[index] = self.update_grossberg_weights(
                    y_value, self.grossberg_weights[index], learning_rate=lr_grossberg)
                grossberg_neuron_out = int(
                    round(self.sum_activation(kohonen_neurons, self.grossberg_weights)))
                good_counter += self.good_count(y_value, grossberg_neuron_out)

                y_pred[idx] = grossberg_neuron_out
                idx += 1

                print(f'{y_value} : {grossberg_neuron_out}')

            print(
                f'Success training {epoch+1} epoch: {int(good_counter/len(self.y_values) * 100)}%')
            print(epoch)
            if plot_3d:
                fig_3d.suptitle('Epoch %d' % epoch)
                self.live_plot_3d(ax3d, y_pred)
            else:
                fig.suptitle('Epoch %d' % epoch)
                self.live_plot(axs, y_pred, iris)

            # axs[0].clear()  # clear the line
            # axs[1].clear()  # clear the line
            # # axs[0].scatter(X[:, 0], X[:, 1], marker='o', c=y)

            # axs[0].scatter(self.X_values[:, 0], self.X_values[:, 1], marker='x',
            #                c=self.y_values, label='kohonen')
            # axs[0].scatter(self.kohonen_weights[:, 0], self.kohonen_weights[:, 1],
            #                marker='x', c='r', label='kohonen')
            # axs[0].set_xlabel("kohonen")

            # axs[1].scatter(self.X_values[:, 0], self.X_values[:, 1],  c=y_pred, marker='.',
            #                label='grossberg')
            # axs[1].set_xlabel("grossberg")

            # plt.pause(0.001)

    def live_plot(self, axs, y_pred, iris):
        axs[0].clear()  # clear the line
        axs[1].clear()  # clear the line

        col = 1
        if (iris):
            col = 2
        axs[0].scatter(self.X_values[:, 0], self.X_values[:, col], marker='o',
                       c=self.y_values, label='kohonen')
        axs[0].scatter(self.kohonen_weights[:, 0], self.kohonen_weights[:, col],
                       marker='x', c='r', label='kohonen')
        axs[0].set_xlabel("kohonen")

        axs[1].scatter(self.X_values[:, 0], self.X_values[:, col],  c=y_pred, marker='.',
                       label='grossberg')

        axs[1].set_xlabel("grossberg")

        plt.pause(0.001)

    def live_plot_3d(self, ax, y_pred):
        ax[0].clear()
        ax[1].clear()
        ax[0].scatter(self.X_values[:, 0], self.X_values[:, 1], self.X_values[:, 2], marker='o',
                      c=self.y_values, label='kohonen')
        ax[0].scatter(self.kohonen_weights[:, 0], self.kohonen_weights[:, 1], self.kohonen_weights[:, 2],
                      marker='x', c='r')
        ax[0].set_xlabel("kohonen")

        ax[1].scatter(X[:, 0], X[:, 1], X[:, 2], marker='o', c=y)
        ax[1].scatter(self.X_values[:, 0], self.X_values[:, 1],
                      self.X_values[:, 2],  c=y_pred, marker='.')
        ax[1].set_xlabel("grossberg")

        plt.pause(0.0001)
    # work network on test values

    def evaluate(self, X_values, y_values):
        self.X_values = X_values
        self.y_values = y_values

        good_counter = 0
        guesses = np.zeros(len(y_values))
        idx = 0
        for x_vector, y_value in zip(self.X_values, self.y_values):
            kohonen_neurons = []

            for w_iter in range(len(self.kohonen_weights)):
                kohonen_neurons.append(self.calculate_evklid_way(
                    x_vector, self.kohonen_weights[w_iter]))

            neuron_min = min(kohonen_neurons)
            index = kohonen_neurons.index(neuron_min)

            for i in range(len(kohonen_neurons)):
                if i == index:
                    kohonen_neurons[i] = 1
                else:
                    kohonen_neurons[i] = 0

            grossberg_neuron_out = int(
                round(self.sum_activation(kohonen_neurons, self.grossberg_weights)))
            guesses[idx] = grossberg_neuron_out
            good_counter += self.good_count(y_value, grossberg_neuron_out)
            idx += 1
        print(
            f'Success evaluate: {int(good_counter/len(self.y_values) * 100)}%')
        plot_accuracy(y_values, guesses)


def plot_accuracy(targets, guesses):
    fig = plt.figure(2)
    fig.suptitle('Επιτυχείς/Ανεπιτυχείς κατανομές')

    ax = fig.add_subplot(1, 1, 1)
    # mple teleies: pragmatikoi stoxoi (y_test)
    plt.scatter(range(len(targets)), targets, marker='o', color='b')
    # kokkinoi kykloi: exwdos (predictions)
    plt.scatter(range(len(guesses)), guesses, marker='.', color='r')
    plt.xlabel("protypo")
    plt.ylabel("exodos (r) / stoxos (b)")
    plt.show()


def iris():
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


# work neural network
if __name__ == '__main__':
    print('---------------------1.Load Data---------------------')
    file = input("Δώσε input file (a, b, c, d, ii_a, ii_b, iris): ") or 'a'
    is3d = file.__contains__("ii_")
    isIris = file.__contains__("iris")
    if (isIris):
        is3d = int(input("Δώσε 1 για εμφάνιση αποτελεσμάτων σε 3D: ") or 0) == 1
        X, y = iris()
    else:
        if(is3d):
            is3d = True
            d = 3
        input_file = 'data_package_%s.csv' % file
        df = pd.read_csv(input_file, header=0)
        df = df._get_numeric_data()
        X = df.values

        targets_file = 'data_package_values_%s.csv' % file
        targets_df = pd.read_csv(targets_file, header=0)
        targets_df = targets_df._get_numeric_data()
        y = targets_df.values
    k = int(input("Give kohonen neurons "))
    g = int(input("Give grossberg neurons "))
    epochs = int(input("Give epochs "))
    # X: values, y: targets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    net = Counter_Propagation(X_test, y_test, kohonen_neurons=k,
                              grossberg_neurons=g, len_x_vector=len(X_test[0]))

    t_start = time.perf_counter()
    net.fit(lr_kohonen=0.7, lr_grossberg=0.1,
            epochs=epochs, plot_3d=is3d, iris=isIris)
    t_stop = time.perf_counter()

    print(f"Time of fit: {round(t_stop - t_start, 3)}")

    # testing on synthetic values

    # Assign each datapoint to its predicted cluster
    print("Evaluate")
    net.evaluate(X_test, y_test)
