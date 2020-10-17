import numpy as np
from tqdm import trange
from ml.core.plot import Animator, plot_regression_line
from ml.core.losses import rss
from ml.core.metrics import mse, accuracy, euclidean_distance
from ml.core.regularization import l2_penalty
from ml.core.math import mode

class LinearRegression:
    def __init__(self, method='analytic', regularization_penalty=0, eval_metric=None):
        self.W = None
        self.b = None
        self.method = method
        self.regularization_penalty = regularization_penalty
        self.eval_metric = eval_metric

    def fit(self, X, y, **kwargs):
        try:
            y.shape[1]
        except IndexError:
            y = y.reshape(-1, 1)
        if self.method[0] == 'a':
            self._analytic_method(X, y)
        elif self.method[0] == 'g':
            np.random.seed(42)
            self.W = np.random.normal(loc=0, scale=0.1, size=(X.shape[1]))
            self.b = 0
            return self._gradient_descent_runner(X, y, **kwargs)

    def predict(self, X):
        return np.dot(X, self.W) + self.b

    def evaluate(self, X, y):
        if len(y.shape) > 1:
            y = y.reshape(1, -1)
        return self.eval_metric(y, self.predict(X))

    def _analytic_method(self, X, y):
        if X.shape[1] == 1:
            self._analytic_univariate(X, y)
        else:
            self._analytic_multivariate(X, y)

    def _analytic_univariate(self, X, y):
        y_mean = np.mean(y, axis=0)
        X_mean = np.mean(X, axis=0)

        SS_xy = np.sum((X - X_mean) * (y - y_mean), axis=0)
        SS_xx = np.sum(np.square(X - X_mean), axis=0)

        self.W = SS_xy / SS_xx
        self.b = (y_mean - np.dot(X_mean, self.W))[0]

    def _analytic_multivariate(self, X, y):
        bias = np.ones((X.shape[0],1))
        X = np.concatenate([bias, X], axis=1)

        W = np.transpose(X).dot(X)
        penalty = np.zeros_like(W)
        np.fill_diagonal(penalty, self.regularization_penalty)
        penalty[0, 0] = 0
        W += penalty
        W = np.linalg.inv(W).dot(np.transpose(X))
        W = np.dot(W, y)
        self.W = W[1:].flatten()
        self.b = W[0][0]

    def _compute_error(self, X, y):
        N = len(y)
        y_hat = (np.dot(X, self.W) + self.b).reshape(-1, 1)
        totalError = (1/N) * rss(y, y_hat)
        return totalError

    def _step_gradient(self, X, y, learning_rate, batch_size):
        N = len(y)
        if batch_size is None:
            batch_size = N
        batch_size = min(batch_size, N)

        for b in range(0, N, batch_size):
            X_batch = X[b:b + batch_size]
            y_batch = y[b:b + batch_size]
            bs = len(X_batch)
            bs_div = max(1, batch_size % N)

            y_hat = (np.dot(X_batch, self.W) + self.b).reshape(-1, 1)

            w_penalty = l2_penalty(self.regularization_penalty, self.W) / bs_div
            dW = -2/bs * (np.sum(X_batch * (y_batch - y_hat), axis=0) + w_penalty)
            b_penalty = l2_penalty(self.regularization_penalty, self.b) / bs_div
            db = -2/bs * (np.sum(y_batch - y_hat) + b_penalty)

            self.W -= (dW * learning_rate) / bs_div
            self.b -= (db * learning_rate) / bs_div

    def _gradient_descent_runner(
        self, X, y, learning_rate=0.1, batch_size=None, epochs=10,
        tool=None, verbose=False, return_metrics=False
    ):
        metrics = []
        old_error = self._compute_error(X, y)

        for _ in range(epochs):
            shuffle = np.random.permutation(len(X))
            X, y = X[shuffle], y[shuffle]
            self._step_gradient(X, y, learning_rate, batch_size)
            error = self._compute_error(X, y)
            if tool and abs(old_error - error) < tool:
                break
            old_error = error
            metrics.append(error)
        if return_metrics:
            return metrics

class PolynomialLinearRegression(LinearRegression):
    def __init__(self, p, **kwargs):
        super(PolynomialLinearRegression, self).__init__(**kwargs)
        self.eval_metric = None
        self.p = p

    def fit(self, X, y, **kwargs):
        X = np.concatenate([np.power(X, i) for i in range(1, self.p + 1)], axis=1)
        super().fit(X, y, **kwargs)

    def predict(self, X):
        X = np.concatenate([np.power(X, i) for i in range(1, self.p + 1)], axis=1)
        return super().predict(X)

class LogisticRegression:
    def __init__(self, limiar=0.5, penalty=0):
        self.W = None
        self.b = None
        self.limiar = limiar
        self.penalty = penalty
        self.classes = None

    def fit(self, X, y, learning_rate=0.1, epochs=10, batch_size=None, tool=None, return_errors=False):
        y = y.reshape(-1, 1)
        self._multiclass = np.unique(y).shape[0] > 2
        if not self._multiclass:
            self.W = np.random.normal(loc=0, scale=0.1, size=(X.shape[1]))
            self.b = 0
            errors = self._gradient_descent_runner(X, y, learning_rate, epochs, tool, batch_size)
            if return_errors:
                errors = np.pad(errors, (0, epochs - len(errors)), mode='minimum')
                return errors
        else:
            self.W = []
            self.b = []
            self.classes = dict()
            errors = []
            for i, c in enumerate(np.unique(y)):
                clf = RegressaoLogisticaGD(self.limiar, self.penalty)
                err = clf.fit(
                    X, (y == c),
                    learning_rate=learning_rate, epochs=epochs,
                    batch_size=batch_size, tool=tool, return_errors=True
                )
                errors.append(err)
                self.W.append(clf.W)
                self.b.append(clf.b)
                self.classes[i] = c

            errors = np.array(errors).mean(axis=0)
            self.W = np.array(self.W)
            self.b = np.array(self.b)
            if return_errors:
                return errors

    def predict(self, X):
        if not self._multiclass:
            y_hat = (np.dot(X, self.W) + self.b).reshape(-1, 1)
            y_hat = 1 / (1 + np.exp(-y_hat))
            return y_hat > self.limiar
        else:
            y_hat = []
            for W, b in zip(self.W, self.b):
                y_ = (np.dot(X, W) + b).reshape(-1, 1)
                y_ = 1 / (1 + np.exp(-y_))
                y_hat.append(y_)
            y_hat = np.array(y_hat)
            y_hat = np.argmax(y_hat, axis=0)
            map_classes = np.vectorize(lambda x: self.classes[x])
            return map_classes(y_hat)

    def _compute_error(self, X, y):
        N = len(y)
        y_hat = (np.dot(X, self.W) + self.b).reshape(-1, 1)
        y_hat = 1 / (1 + np.exp(-y_hat))
        totalError = (1/N) * rss(y, y_hat)
        return totalError

    def _step_gradient(self, X, y, learning_rate, batch_size):
        N = len(y)
        if batch_size is None:
            batch_size = N
        batch_size = min(batch_size, N)

        shuffle = np.random.permutation(len(X))
        X, y = X[shuffle], y[shuffle]

        for b in range(0, N, batch_size):
            X_batch = X[b:b + batch_size]
            y_batch = y[b:b + batch_size]
            bs = len(X_batch)
            bs_div = max(1, bs % N)

            y_hat = (np.dot(X, self.W) + self.b).reshape(-1, 1)
            y_hat = 1 / (1 + np.exp(-y_hat))

            w_penalty = l2_penalty(self.penalty, self.W) / bs_div
            dW = -1/bs * (np.sum(X_batch * (y_batch - y_hat), axis=0) + w_penalty)

            b_penalty = l2_penalty(self.penalty, self.b) / bs_div
            db = -1/bs * (np.sum(y_batch - y_hat) + b_penalty)

            self.W -= (dW * learning_rate) / bs_div
            self.b -= (db * learning_rate) / bs_div

    def _gradient_descent_runner(
        self, X, y, learning_rate, epochs, tool, batch_size
    ):
        old_error = self._compute_error(X, y)
        errors = []
        for epoch in range(epochs):
            self._step_gradient(X, y, learning_rate, batch_size)
            error = self._compute_error(X, y)
            if tool and abs(old_error - error) < tool:
                break
            old_error = error
            errors.append(error)
        return errors

class GaussianNaiveBayes:
    def __init__(self):
        self.mean = None
        self.var = None
        self.prior = None
        self.classes = None

    def fit(self, X, y):
        self.mean = dict()
        self.var = dict()
        self.prior = dict()

        self.classes = np.unique(y)
        for c in self.classes:
            y_ = y == c
            y_ = y_.flatten()
            X_ = X[y_]

            self.mean[c] = np.mean(X_, axis=0)
            self.var[c] = np.var(X_, axis=0)

            self.prior[c] = np.mean(y_)

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.classes.shape[0]))
        for i, c in enumerate(self.classes):
            pxc = self.prior[c]

            for j in range(X.shape[1]):
                x_j = X[:, j]
                a = (1 / (np.sqrt(2 * np.pi * self.var[c][j])))
                exp = np.exp(-(np.square((x_j - self.mean[c][j]))) / self.var[c][j])

                pxc *= a * exp
            probs[:, i] = pxc

        max_prob = np.argmax(probs, axis=1)
        map_classes = np.vectorize(lambda x: self.classes[x])
        return map_classes(max_prob)

class GaussianQuadraticDiscriminant:
    def __init__(self):
        self.mean = None
        self.covariance = None
        self.classes = None

    def fit(self, X, y):
        self.mean = dict()
        self.covariance = dict()
        self.classes = np.unique(y)

        for c in self.classes:
            y_ = y == c
            y_ = y_.flatten()
            X_ = X[y_]

            self.mean[c] = np.mean(X_, axis=0)
            self.covariance[c] = np.cov(X_.T)

    def predict(self, X):
        probs = np.zeros((X.shape[0], self.classes.shape[0]))

        for i in range(X.shape[0]):
            for j, c in enumerate(self.classes):
                d = X[i, :] - self.mean[c]

                dp = (-1/2) * np.dot(np.dot(d.T, np.linalg.inv(self.covariance[c])), d)
                sq = 1 / np.sqrt((2 * np.pi ** self.classes.shape[0]) * np.linalg.det(self.covariance[c]))
                probs[i, j] = sq * np.exp(dp)

        max_prob = np.argmax(probs, axis=1)
        map_classes = np.vectorize(lambda x: self.classes[x])
        return map_classes(max_prob)

class MLP:
    def __init__(self, inp, units, reset=False):
        self.inp = inp
        self.units = units
        self.reset = reset

        self.H = np.random.randn(self.inp, self.units)
        self.O = np.random.randn(self.units, 1)

    def _sigmoid(self, y):
        return 1 / (1 + np.exp(-y))

    def _sigmoid_derivative(self, y):
        return self._sigmoid(y) * (1 - self._sigmoid(y))

    def _get_classess(y_pred):
        return np.greater(y_pred, 0.5).astype(int).flatten()

    def fit(self, X, y, learning_rate=1e-2, epochs=10, return_losses=False, verbose=None):
        if self.reset:
            self.H = np.random.randn(self.inp, self.units)
            self.O = np.random.randn(self.units, 1)
        y = y.reshape(-1, 1)
        losses = []
        for e in range(epochs):
            h = X @ self.H
            h_A = self._sigmoid(h)
            o = h_A @ self.O
            o_A = self._sigmoid(o)

            grad_o_A = - (np.divide(y, o_A) - np.divide(1 - y, 1 - o_A))
            grad_o = grad_o_A * self._sigmoid_derivative(o)
            grad_O = h_A.T @ grad_o

            grad_h_A = grad_o_A @ self.O.T
            grad_h = grad_h_A * self._sigmoid_derivative(h)
            grad_H = X.T @ grad_h

            self.H -= grad_H * learning_rate
            self.O -= grad_O * learning_rate

            loss = mse(o_A, y)
            losses.append(loss)
            if verbose and e % verbose == 0:
                acc = accuracy(y, self._get_classess(o_A))
                print('Epoch {}: {} loss / {} accuracy'.format(e, loss, acc))
        if return_losses:
            return losses

    def predict(self, X):
        h = X @ self.H
        h_A = self._sigmoid(h)
        o = h_A @ self.O
        o_A = self._sigmoid(o)
        return np.greater(o_A, 0.5).astype(int).flatten()

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.features = X
        self.classes = y

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for i, r in enumerate(X):
            distances = [*map(lambda x: euclidean_distance(r, x), self.features)]
            nearest = np.argsort(distances)
            nearest = self.classes[nearest[:self.k]]
            y_pred[i] = mode(nearest)
        return y_pred

class KMeans():
    def __init__(self, n_clusters=8, max_iter=300, tool=0.0001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tool = tool
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None

    def fit(self, X):
        self.cluster_centers_ = self._calculate_initial_centers(X)
        self.labels_ = self._all_nearest_centroids(X)
        old_inertia = self._inertia(X)

        for _ in range(self.max_iter):
            self.cluster_centers_ = self._update_centroids(X)
            self.labels_ = self._all_nearest_centroids(X)
            self.inertia_ = self._inertia(X)
            if np.abs(old_inertia - self.inertia_) < self.tool:
                break
            old_inertia = self.inertia_
        return self

    def predict(self, X):
        return self._all_nearest_centroids(X)

    def _calculate_initial_centers(self, dataset):
        return np.random.uniform(
            dataset.min(axis=0),
            dataset.max(axis=0),
            (self.n_clusters, dataset.shape[1])
        )

    def _nearest_centroid(self, a):
        distances = np.zeros(self.cluster_centers_.shape[0])

        for i, centroid in enumerate(self.cluster_centers_):
            distances[i] = euclidean_distance(a, centroid)

        return np.argmin(distances)

    def _all_nearest_centroids(self, dataset):
        nearest_indexes = np.zeros(dataset.shape[0])

        for i, a in enumerate(dataset):
            nearest_indexes[i] = self._nearest_centroid(a)

        return nearest_indexes


    def _inertia(self, dataset):
        inertia = 0
        for i, centroid in enumerate(self.cluster_centers_):
            dataframe = dataset[self.labels_ == i, :]
            for a in dataframe:
                inertia += np.square(euclidean_distance(a, centroid))

        return inertia

    def _update_centroids(self, dataset):
        for i, centroid in enumerate(self.cluster_centers_):
            dataframe = dataset[self.labels_ == i, :]
            if dataframe.shape[0] != 0:
                self.cluster_centers_[i] = np.mean(dataframe, axis=0)
        return self.cluster_centers_
