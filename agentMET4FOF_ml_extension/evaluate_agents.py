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
from .util.calc_auroc import calc_all_scores, calc_auroc_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

class EvaluateSupervisedUncAgent(ML_BaseAgent):
    def init_parameters(self, figsize=(10,5)):
        self.figsize = figsize

    def on_received_message(self, message):
        """
        First dim: mean (0) and std (1) of supervised predictions
        Second dim: num examples (size of dataset)
        """

        xs = message["data"]["quantities"]
        ys = message["data"]["target"]

        if isinstance(xs,dict):
            evaluations = {}
            plots = []
            for key in xs.keys():
                evaluation, plot = self.apply_evaluate_method(xs[key],ys[key], key=key)
                evaluations.update({key:evaluation})
                plots.append(plot)
        else:
            evaluations, plots = self.apply_evaluate_method(xs,ys)

        self.send_plot(plots)
        self.send_output(evaluations, channel="default")

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
        new_fig = self.plot_uncertainty_calibration(x=x_mean,ux=x_std, y=y_temp,
                                                    title="Axis "+key+" RMSE: "+str(round(rmse,2)) + " UNC: "+str(round(avg_unc,2)),
                                                    figsize=self.figsize)

        return {"rmse":rmse, "avg_unc":avg_unc}, new_fig

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



class EvaluateAUROCAgent(ML_BaseAgent):
    """
    Takes in BAE mean-var NLL of test inlier and ood and squashes them to 1 dimension as the anomaly score.
    Calculates the AUROC for the inlier and ood scores.

    Should be used in tandem with `compute_mean_var` agent with return_dict =True
    """
    def init_parameters(self, figsize=(10,5)):
        self.figsize = figsize

    def on_received_message(self, message):
        channel = message["channel"]

        if channel == "test":
            msg_data = message["data"]
            x_test = msg_data["quantities"]["test"]
            x_ood = msg_data["quantities"]["ood"]
            plots = []
            for key in list(x_test.keys()):
                x_test_, x_ood_ = self.sum_mean_var(x_test[key],x_ood[key])
                auroc = calc_auroc_score(x_test_, x_ood_)

                plots.append(self.plot_histogram(x_test_,x_ood_, title="AUROC-"+key+":"+str(round(auroc,2))))
                self.send_output({"AUROC-"+key: auroc}, channel=channel)
            self.send_plot(plots)


    def sum_mean_var_(self, bae_test):
        while len(bae_test.shape)>1:
            bae_test = bae_test.sum(-1)
        return bae_test

    def sum_mean_var(self, *bae_test):
        return (self.sum_mean_var_(bae_test_i) for bae_test_i in bae_test)

    def plot_histogram(self, x_test,x_ood, title=""):
        fig = plt.figure()
        plt.hist(x_test)
        plt.hist(x_ood)
        plt.legend(["TEST","OOD"])
        plt.title(title)
        return fig






















