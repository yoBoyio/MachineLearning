import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.activation_function
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        #samples = rows , feature= columns
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # y must be only 0 or 1
        y_ = np.array([1 if i > 0 else 0 for i in y])
        # epochs
        for _ in range(self.n_iters):
            # for each Xi
            for index, xi in enumerate(X):
                #  Predict Xi * Wi + B
                linear_output = np.dot(xi, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                # Δw := a*(Yi - pYi)*Xi
                update = self.lr * (y_[index] - y_predicted)
                self.weights += update * xi
                self.bias += update

    def predict(self, X):
        # Xi * Wi + B
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

# activation function
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
