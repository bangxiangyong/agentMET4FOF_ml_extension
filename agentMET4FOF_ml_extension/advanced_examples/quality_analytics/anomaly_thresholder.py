from statsmodels.distributions import ECDF
import numpy as np

class AnomalyThreshold():
    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, x_train, y_train=None):
        self.ecdf_model, x_range, ecdf_vals = self.fit_ecdf(x_train)
        self.x_threshold = x_range[np.argwhere(ecdf_vals >= self.threshold)[0]]
        return self

    def transform(self, x_test):
        return np.piecewise(x_test, [x_test <= self.x_threshold, x_test > self.x_threshold], [0, 1])

    def fit_ecdf(self, x):
        ecdf_model = ECDF(x)
        x_min, x_max = np.min(x), np.max(x)
        half = (np.abs(x_max) + np.abs(x_min))/2
        x_range = np.linspace(x_min-half, x_max+half, 500)
        ecdf_vals = ecdf_model(x_range)
        return ecdf_model, x_range, ecdf_vals

    def plot_ecdf(self, x,  ax, complementary=False):
        _, x_range, ecdf_vals = self.fit_ecdf(x)
        if complementary:
            ax.plot(x_range, ecdf_vals * -1 + 1)
        else:
            ax.plot(x_range, ecdf_vals)
