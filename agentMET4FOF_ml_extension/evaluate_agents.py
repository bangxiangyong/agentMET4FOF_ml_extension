import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
from baetorch.baetorch.util.data_model_manager import DataModelManager
from baetorch.baetorch.util.seed import bae_set_seed
from .datastreams import IRIS_Datastream, BOSTON_Datastream
import inspect
import functools
import pandas as pd
# Datastream Agent
from .ml_agents import ML_BaseAgent
from .util.calc_auroc import calc_all_scores
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class EvaluateSupervisedUncAgent(ML_BaseAgent):
    def on_received_message(self, message):
        """
        First dim: mean (0) and std (1) of supervised predictions
        Second dim: num examples (size of dataset)
        """

        xs = message["data"]["quantities"]
        ys = message["data"]["target"]

        if isinstance(xs,dict):
            evaluations = [self.apply_evaluate_method(xs[key],ys[key], key=key) for key in xs.keys()]
        else:
            evaluations = self.apply_evaluate_method(xs,ys)

        self.send_plot(evaluations)

    def apply_evaluate_method(self, x,y, key=""):
        # do something with x and y
        x_mean = x[0]
        x_std = x[1]
        if isinstance(y, pd.DataFrame):
            y_temp = y.values[:,0]
        else:
            y_temp = y[:,0]
        argsorted = np.argsort(y_temp)
        y_temp = y_temp[argsorted]
        x_mean = x_mean[argsorted]
        x_std = x_std[argsorted]

        # compute statistics
        rmse = self.rmse(y_temp, x_mean)
        avg_unc = self.avg_unc(x_std)
        # picp = self.picp(y_temp, x_mean, x_std)

        # plots
        new_fig = self.plot_uncertainty_calibration(x=x_mean,ux=x_std, y=y_temp, title="Axis "+key+" RMSE: "+str(round(rmse,2)) + " UNC: "+str(round(avg_unc,2)))
        return new_fig

    def rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))

    def avg_unc(self, x_std):
        return np.mean(x_std)

    # def picp(self, y_true, y_pred):
    #     return np.sqrt(mean_squared_error(y_true,y_pred))

    def plot_uncertainty_calibration(self, x, ux, y, title="", figsize=(10,5)):
        fig, (ax1,ax2) = plt.subplots(1,2,figsize=figsize)

        ax1.plot(y, x, color="blue")
        ax1.fill_between(y,x+2*ux, x-2*ux, alpha=0.55, color="orange")
        ax1.set_ylabel("Y Predicted")
        ax1.set_xlabel("Y True")
        ax1.set_ylim(0,100)

        ax2.plot(y, ux, color="blue")
        ax2.set_ylabel("Y Uncertainty")
        ax2.set_xlabel("Y True")
        ax2.set_ylim(0, 100)

        fig.suptitle(title, y=1.00)
        fig.tight_layout()
        return fig



























