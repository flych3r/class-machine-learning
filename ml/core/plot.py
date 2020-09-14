import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from ml.core.metrics import confusion_matrix


def config_plot():
    display.set_matplotlib_formats('jpg', quality=94)
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.dpi'] = 120
    plt.rcParams['figure.figsize'] = 12, 8


def plot_regression_line(X, y, lr):
    # plotting the actual points as scatter plot
    plt.scatter(
        X[:, 0], y, color = "m",
        marker = "o", s = 30
    )

    # predicted response vector
    y_pred = lr.predict(X)
    y_pred = y_pred.flatten()
    # plotting the regression line
    plt.plot(X[:, 0], y_pred, color = "g")

    # putting labels
    plt.xlabel('X')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def plot_lines(lines, labels=None, title=None, xlabel=None, ylabel=None):
    size = len(lines)
    if labels is None:
        labels = [None] * size
    lim = 1
    for ln, lb in zip(lines, labels):
        plt.plot(ln, label=lb)
        lim = max(lim, len(ln))
    plt.legend()
    plt.xlim(0, lim - 1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot()


def plot_confusion_matrix(X, y, clf):
    y_pred = clf.predict(X)
    conf = confusion_matrix(y, y_pred)

    lim = np.max(conf) / 2

    fig, ax = plt.subplots()

    im = ax.matshow(conf, cmap='Reds')
    fig.colorbar(im)
    for (i, j), z in np.ndenumerate(conf):
        ax.text(j, i, z, ha='center', va='center', color='white' if z > lim else 'black')

    plt.grid(False)


def plot_boundaries(X, y, clf):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    try:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    except:
        raise ValueError('To plot the boundaries, the classifier must be fitted using only 2 features')

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    for c in np.unique(y):
        mask = (y == c).flatten()
        plt.scatter(X[mask, 0], X[mask, 1], label=c, edgecolors='k', cmap=plt.cm.Paired)
    plt.legend()
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.grid(False)


class Animator:
    def __init__(self, xlabel=None, ylabel=None, legend=None, title=None,
                 xlim=None, ylim=None, xscale='linear', yscale='linear',
                 fmts=None, nrows=1, ncols=1, figsize=(15, 10)):
        """Incrementally plot multiple lines."""
        if legend is None:
            legend = []
        if title is None:
            title = ''
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda to capture arguments
        self.config_axes = lambda: self.set_axes(
            xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, xlabel, ylabel, xlim, ylim, xscale, yscale, legend, title):
        """Set the axes for matplotlib."""
        self.axes[0].set_xlabel(xlabel)
        self.axes[0].set_ylabel(ylabel)
        self.axes[0].set_xscale(xscale)
        self.axes[0].set_yscale(yscale)
        self.axes[0].set_xlim(xlim)
        self.axes[0].set_ylim(ylim)
        if legend:
            self.axes[0].legend(legend)
        if title:
            self.axes[0].set_title(title)
        self.axes[0].grid()

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not hasattr(x, '__len__'):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = ['-'] * n
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
