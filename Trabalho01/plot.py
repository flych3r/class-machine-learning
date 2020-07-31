import matplotlib.pyplot as plt
import numpy as np
from IPython import display


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