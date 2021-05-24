import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.activation_function
        self.weight = None
        self.bias = None

    def fit(self, X, y, plot=False, plot_single=False):
        #samples = rows , feature= columns
        n_samples, n_features = X.shape

        # init weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # y must be only 0 or 1
        y_ = np.array([1 if i > 0 else 0 for i in y])
        # epochs
        if plot:
            fig, axes = plt.subplots(1,3)
        if plot_single:
            fig_single, ax = plt.subplots(1)
            
            
        for epoch in range(self.n_iters):
            y_pred = np.zeros(y.shape)
    
            # for each Xi
            for index, xi in enumerate(X):
                #  Predict Xi * Wi + B
                linear_output = np.dot(xi, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                # Δw := a*(Yi - pYi)*Xi
                update = self.lr * (y_[index] - y_predicted)
                self.weights += update * xi
                self.bias += update
                y_pred[index] = y_predicted
            if(epoch%10==0): #update every 10 epochs
                if plot:
                    self.live_plot(axes,X,y,self.weights, self.bias, y_pred, epoch)
                if plot_single:
                    self.live_plot_single(ax, targets=y, predictions=y_pred, epoch=epoch)

    def predict(self, X):
        # Xi * Wi + B
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    # activation function
    def activation_function(self, x):
        return np.where(x > 0, 1, 0)

    def live_plot(self,axes,X,y,w,b,y_pred,epoch):
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].scatter(X[:, 0], X[:, 1], marker='o', c=y)
        axes[1].scatter(X[:, 0], X[:, 1], marker='x', c=y_pred)
        axes[2].scatter(range(len(y_pred)), y_pred,marker='x', c=y_pred)
        axes[2].set_xlabel('Epoch %d' % epoch)

        x_intercept = -b/w[1]
        y_intercept = -b/w[0]

        axes[1].plot([0, x_intercept], [y_intercept,0], 'k')

        ymin = np.amin(X[:, 1])
        ymax = np.amax(X[:, 1])
        axes[1].set_ylim([ymin-1, ymax+1])
        axes[0].set_ylim([ymin-1, ymax+1])

        axes[0].set_xlabel("protypo")
        axes[0].set_ylabel("exodos ")
        axes[1].set_xlabel("protypo")
        axes[1].set_ylabel("exodos ")
        plt.pause(0.0001)

    def live_plot_single(self, ax, targets, predictions, epoch):
        ax.clear()
        ax.scatter(range(len(targets)), targets, marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
        ax.scatter(range(len(predictions)), predictions, marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
        ax.set_xlabel('protypo Epoch %d'% epoch)
        ax.set_ylabel("exodos (r) / stoxos (b)")
        plt.pause(0.0001)