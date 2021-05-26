import numpy as np
import matplotlib.pyplot as plt


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self.activation_function
        self.weight = None
        self.bias = None

    def fit(self, X, y, plot=False, plot_single=False, plot_3d=False):
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
        if plot_3d:
            fig_3d = plt.figure(figsize=(12,4))
            gs = fig_3d.add_gridspec(1, 3)
            ax_3d_0 = fig_3d.add_subplot(gs[0,0], projection='3d')
            ax_3d_1= fig_3d.add_subplot(gs[0,1], projection='3d')
            ax_3d_2= fig_3d.add_subplot(gs[0,2])
            ax3d=[ax_3d_0,ax_3d_1,ax_3d_2]
            
            
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
            if(epoch%1==0): #update every 10 epochs
                if plot:
                    fig.suptitle("Epoch %d" %epoch)
                    self.live_plot(axes,X,y,self.weights, self.bias, y_pred)
                if plot_single:
                    fig_single.suptitle("Epoch %d" %epoch)
                    self.live_plot_single(ax, targets=y, predictions=y_pred)
                if plot_3d:
                    fig_3d.suptitle("Epoch %d" %epoch)
                    self.live_plot_3d(ax3d, X, y, y_pred, self.weights, self.bias)

    def predict(self, X):
        # Xi * Wi + B
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    # activation function
    def activation_function(self, x):
        return np.where(x > 0, 1, 0)

    def get_weights_bias(self):
        return [self.weights, self.bias]

    def live_plot(self,axes,X,y,w,b,y_pred):
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        axes[0].scatter(X[:, 0], X[:, 1], marker='o', c=y)
        axes[1].scatter(X[:, 0], X[:, 1], marker='x', c=y_pred)
        axes[2].scatter(range(len(y_pred)), y_pred,marker='x', c=y)

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

    def live_plot_3d(self, ax, X, y, y_pred, w, b):
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].scatter(X[:, 0], X[:, 1],X[:, 2], marker='x', c=y)        
        ax[1].scatter(X[:, 0], X[:, 1],X[:, 2], marker='x', c=y_pred)
        ax[2].scatter(range(len(y_pred)), y_pred,marker='x', c=y)

        w1 = w[0] #a
        w2 = w[1] #b
        w3 = w[2] #c
        # Diaxoristiko epipedo: ax + by + cz = d
        a,b,c,d = w1,w2,w3,b
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        ax[1].set_xlim([x_min-0.2, x_max+0.2])
        x = np.linspace(x_min, x_max, 100)
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])
        ax[1].set_ylim([y_min-0.2, y_max+0.2])
        z_min = np.amin(X[:, 2])
        z_max = np.amax(X[:, 2])
        ax[1].set_zlim([z_min+0.2, z_max+0.2])
        y = np.linspace(y_min, y_max, 100)
        Xs,Ys = np.meshgrid(x,y)
        Zs = ((d + a*Xs + b*Ys) / c)*(-1)
        ax[1].plot_surface(Xs, Ys, Zs, alpha=0.45)
        plt.pause(0.0001)

    def live_plot_single(self, ax, targets, predictions):
        ax.clear()
        ax.scatter(range(len(targets)), targets, marker='o', color='b') # mple teleies: pragmatikoi stoxoi (y_test)
        ax.scatter(range(len(predictions)), predictions, marker='.', color='r') # kokkinoi kykloi: exwdos (predictions)
        ax.set_xlabel('protypo')
        ax.set_ylabel("exodos (r) / stoxos (b)")
        plt.pause(0.0001)