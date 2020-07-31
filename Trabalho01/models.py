import numpy as np
import time
from plot import Animator, plot_regression_line
from tqdm import trange
from losses import rss
from regularization import l2_penalty

class LinearRegression:
    def __init__(self, method='analytic', regularization_penalty=0, eval_metric=None):
        self.W = None
        self.b = None
        self.method = method
        self.regularization_penalty = 0
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
            y_hat = (np.dot(X_batch, self.W) + self.b).reshape(-1, 1)

            dW = -2/batch_size * np.sum(X_batch * (y_batch - y_hat), axis=0)
            dW -= l2_penalty(self.regularization_penalty, self.W)
            db = -2/batch_size * np.sum(y_batch - y_hat)
            db -= l2_penalty(self.regularization_penalty, self.b)

            self.W -= (dW * learning_rate) / max(1, batch_size % N)
            self.b -= (db * learning_rate) / max(1, batch_size % N)

    def _gradient_descent_runner(
        self, X, y, learning_rate=0.1, batch_size=None, epochs=10,
        tool=None, verbose=False, val_set=None, return_metrics=False
    ):
        metrics = []
        old_error = self._compute_error(X, y)

        epochs = range(epochs)
        if verbose == 1:
            epochs = trange(len(epochs), desc='epoch 0', leave=True)
        elif verbose == 2:
            legend = ['train loss', 'train eval']
            if val_set:
                legend += ['val loss', 'val eval']
            animator = Animator(
                xlabel='epoch', xlim=[1, len(epochs)], ylim=[0, np.max(y) * 2],
                legend=legend, title='Training loss and eval'
            )

        for epoch in epochs:
            self._step_gradient(X, y, learning_rate, batch_size)
            error = self._compute_error(X, y)
            if tool and abs(old_error - error) < tool:
                break
            old_error = error
            train_metrics = [error]
            val_metrics = []
            if verbose:
                if self.eval_metric is not None:
                    train_metrics.append(self.evaluate(X, y))
                if val_set:
                    val_metrics = [self._compute_error(val_set[0], val_set[1])]
                    if self.eval_metric is not None:
                        val_metrics.append(self.evaluate(val_set[0], val_set[1]))
                if verbose == 1:
                    msg = 'epoch {} => {} {}'
                    train_log = 'train({} {})'.format(
                        'loss: {:.5f}'.format(train_metrics[0]),
                        'eval: {:.5f}'.format(train_metrics[1])
                        if len(train_metrics) > 1 else ''
                    )
                    val_log = ''
                    if val_set:
                        val_log = 'val({} {})'.format(
                            'loss: {:.5f}'.format(val_metrics[0]),
                            'eval: {:.5f}'.format(val_metrics[1])
                            if len(val_metrics) > 1 else ''
                        )
                    epochs.set_description(
                        msg.format(
                            epoch + 1,
                            train_log,
                            val_log
                        )
                    )
                    time.sleep(.01)
                    epochs.refresh()
                elif verbose == 2:
                    animator.add(epoch + 1, train_metrics + val_metrics)
            metrics.append((train_metrics, val_metrics))

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
