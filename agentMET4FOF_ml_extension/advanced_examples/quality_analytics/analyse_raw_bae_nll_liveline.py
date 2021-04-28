import os
import pickle

import numpy as np
from scipy.stats import iqr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import matthews_corrcoef, f1_score

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.evaluate_agents import EvaluateAUROCAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_BaseAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent, STRATH_DatastreamAgent, Liveline_DatastreamAgent
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateRegressionClassificationAgent
from agentMET4FOF_ml_extension.util.helper import move_axis, compute_mean_var
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
from statsmodels.distributions.empirical_distribution import ECDF

np.random.seed(100)

import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

# set drift
# moving average
# normalise
# evaluate ood

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


#===========================
# if os.path.exists("dict_raw_nll.p"):
#     dict_nll = pickle.load(open("dict_raw_nll.p","rb"))
#     x_train = dict_nll["quantities"]["train"]["mean"].sum(-1)
#     x_test = dict_nll["quantities"]["test"]["mean"].sum(-1)
#     x_ood = dict_nll["quantities"]["ood"]["mean"].sum(-1)
#     y_train = dict_nll["target"]["train"]
#     y_test = dict_nll["target"]["test"]
#     y_ood = dict_nll["target"]["ood"]
#     y_ood = np.ones(len(y_ood))
#
# anom_thresholder = AnomalyThreshold()
#
# anom_thresholder.fit(x_train)
# # y_pred_train =anom_thresholder.transform(x_train)
# y_pred_test = anom_thresholder.transform(x_test)
# y_pred_ood = anom_thresholder.transform(x_ood)
#
# matplotlib.use('Qt5Agg')
# fig, ax = plt.subplots(1,1)
# anom_thresholder.plot_ecdf(x_train, ax = ax)
# anom_thresholder.plot_ecdf(x_test, ax = ax)
# anom_thresholder.plot_ecdf(x_ood, ax = ax)
# ax.vlines(x=anom_thresholder.x_threshold, ymin=0,ymax=1)
# #===========================
#
# y_true_all = np.concatenate((y_test, y_ood))
# y_pred_all = np.concatenate((y_pred_test, y_pred_ood))
#
# mcc_score = matthews_corrcoef(y_true_all, y_pred_all)
# f1_score_ = f1_score(y_true_all, y_pred_all)

matplotlib.use('Qt5Agg')

if os.path.exists("raw_bae_nll.p"):
    dict_nll = pickle.load(open("raw_bae_nll.p","rb"))["data"]
    x_train = dict_nll["quantities"]["train"].sum(-1)
    x_test = dict_nll["quantities"]["test"].sum(-1)
    x_ood = dict_nll["quantities"]["ood"].sum(-1)
    y_train = dict_nll["target"]["train"]
    y_test = dict_nll["target"]["test"]
    y_ood = dict_nll["target"]["ood"]
    y_ood = np.ones(len(y_ood))

anomaly_thresholders = [AnomalyThreshold().fit(x_bae_sample) for x_bae_sample in x_train]

x_min, x_max = np.min(x_train),np.max(x_train)
half = (x_max+x_min)/2
x_range = np.linspace(x_min-half,x_max+half,1000)
# fig, ax = plt.subplots(1,1)
# for id, model in enumerate(anomaly_thresholders):
#     y_sample_out = model.ecdf_model(x_range)
#     ax.plot(x_range,y_sample_out)
    # model.plot_ecdf(x=x_train[id], ax=ax, complementary=False)

ecdf_samples = np.array([model.ecdf_model(x_range) for model in anomaly_thresholders])
ecdf_samples_aleatoric = np.mean(ecdf_samples*(1-ecdf_samples),axis=0)

ecdf_mean = np.mean(ecdf_samples,axis=0)
ecdf_std = np.std(ecdf_samples,axis=0)
# ecdf_lb = ecdf_mean-2*ecdf_std
# ecdf_ub = ecdf_mean+2*ecdf_std

ecdf_ub = np.quantile(ecdf_samples, 0.95, axis=0)
ecdf_lb = np.quantile(ecdf_samples, 0.05, axis=0)
ecdf_median = np.quantile(ecdf_samples, 0.5, axis=0)
ecdf_iqr = iqr(ecdf_samples,axis=0)
ecdf_total_unc = ecdf_iqr+ ecdf_samples_aleatoric

# fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1)
fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(x_range, ecdf_median)
ax1.fill_between(x_range, ecdf_ub,ecdf_lb, alpha=0.5)
ax2.plot(x_range,ecdf_iqr)
ax2.plot(x_range,ecdf_samples_aleatoric)
ax2.plot(x_range,ecdf_total_unc)

ax2.legend(["EPI","ALEA","TOT"])
# ax3.plot(x_range,ecdf_samples_aleatoric)
# ax4.plot(x_range,ecdf_total_unc)

# best_xtimates scatters
x_train_mean = x_train.mean(0)
x_test_mean = x_test.mean(0)
x_ood_mean = x_ood.mean(0)

ax1.scatter(x_train_mean, np.zeros(len(x_train_mean)), alpha=0.1)
ax1.scatter(x_test_mean, np.zeros(len(x_test_mean)), alpha=0.1)
ax1.scatter(x_ood_mean, np.ones(len(x_ood_mean)), alpha=0.1)

# np.quantile(ecdf_samples, 0.95, axis=0)
# np.quantile(ecdf_samples, 0.05, axis=0)

# fit on train


#=======================ECDF PLOTS==================================

def plot_ecdf_bands(x_bae_samples, ax1,ax2, color="b"):
    anomaly_thresholders = [AnomalyThreshold().fit(x_bae_sample) for x_bae_sample in x_bae_samples]

    x_min, x_max = np.min(x_train),np.max(x_train)
    half = (x_max+x_min)/2
    x_range = np.linspace(x_min-half,x_max+half,1000)

    # ECDF PLOTS
    ecdf_samples = np.array([model.ecdf_model(x_range) for model in anomaly_thresholders])
    ecdf_samples_aleatoric = np.mean(ecdf_samples*(1-ecdf_samples),axis=0)

    ecdf_mean = np.mean(ecdf_samples,axis=0)
    ecdf_std = np.std(ecdf_samples,axis=0)

    ecdf_ub = np.quantile(ecdf_samples, 0.95, axis=0)
    ecdf_lb = np.quantile(ecdf_samples, 0.05, axis=0)
    ecdf_median = np.quantile(ecdf_samples, 0.5, axis=0)
    ecdf_iqr = iqr(ecdf_samples,axis=0)
    ecdf_total_unc = ecdf_iqr+ ecdf_samples_aleatoric

    # Actual plots
    ax1.plot(x_range, ecdf_median, color=color)
    ax1.fill_between(x_range, ecdf_ub,ecdf_lb, alpha=0.5, color=color)
    ax2.plot(x_range,ecdf_iqr)
    ax2.plot(x_range,ecdf_samples_aleatoric)
    ax2.plot(x_range,ecdf_total_unc)

    ax2.legend(["EPI","ALEA","TOT"])

    return anomaly_thresholders, x_range, ecdf_median, ecdf_iqr, ecdf_ub, ecdf_lb

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
trained_anomaly_thresholders, x_range, ecdf_median,ecdf_iqr, ecdf_ub, ecdf_lb  = plot_ecdf_bands(x_train, ax1,ax2, color="tab:blue")
plot_ecdf_bands(x_test, ax1,ax2, color="tab:orange")
plot_ecdf_bands(x_ood, ax1,ax2, color="tab:green")

# now assess the p (Acc | uncertain ? )
# f1-score

def get_trained_ecdfs(x_nll_train, anomaly_thresholders):
    ecdf_samples = np.array([model.ecdf_model(x_nll_train) for model in anomaly_thresholders])
    ecdf_median = np.quantile(ecdf_samples, 0.5, axis=0)
    ecdf_iqr = iqr(ecdf_samples,axis=0)

    return ecdf_median, ecdf_iqr

def predict_ecdf_(x_nll_test_i, ecdf_median, ecdf_iqr):
    arg_ = np.argwhere(ecdf_median>=x_nll_test_i)[0]
    return ecdf_median[arg_], ecdf_iqr[arg_]

def predict_ecdf(x_nll_test, ecdf_median, ecdf_iqr):
    joe = np.apply_along_axis(func1d=predict_ecdf_, axis=0, arr=x_nll_test, ecdf_median=ecdf_median, ecdf_iqr=ecdf_iqr)
    return joe

# y_pred_ = predict_ecdf_(x_train_mean, ecdf_median, ecdf_iqr)

def predict_ecdf_(x_nll_test_i, ecdf_median, ecdf_iqr):
    arg_ = np.argwhere(ecdf_median>=x_nll_test_i)[0]
    return ecdf_median[arg_], ecdf_iqr[arg_]

# [predict_ecdf_(x_train_mean_i, ecdf_median=ecdf_median, ecdf_iqr=ecdf_iqr) for x_train_mean_i in x_train_mean]



import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


np.random.seed(1)

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
# gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
ecdf_median_model = IsotonicRegression(out_of_bounds = "clip")
ecdf_ub_model = IsotonicRegression(out_of_bounds = "clip")
ecdf_lb_model = IsotonicRegression(out_of_bounds = "clip")
# gp.fit(X=x_range.reshape(-1,1), y=ecdf_median.reshape(-1,1))

ecdf_median_model.fit(x_range, ecdf_median)
ecdf_ub_model.fit(x_range, ecdf_ub)
ecdf_lb_model.fit(x_range, ecdf_lb)

fitted_outp_median = ecdf_median_model.predict(x_range)
fitted_outp_ub = ecdf_ub_model.predict(x_range)
fitted_outp_lb = ecdf_lb_model.predict(x_range)

# plt.figure()
# plt.plot(x_range, fitted_outp_median, color="tab:blue")
# plt.plot(x_range, fitted_outp_ub, color="tab:blue")
# plt.plot(x_range, fitted_outp_lb, color="tab:blue")
# plt.hlines(y=0.95, xmin=x_range[0], xmax=x_range[-1], color="black")
# plt.plot(x_range, ecdf_median, color="tab:orange")
# plt.plot(x_range, ecdf_ub, color="tab:orange")
# plt.plot(x_range, ecdf_lb, color="tab:orange")

y_pred_train_median = ecdf_median_model.predict(x_train_mean)
y_pred_test_median = ecdf_median_model.predict(x_test_mean)
y_pred_ood_median = ecdf_median_model.predict(x_ood_mean)

# y_pred_train_binary =

y_pred_train_iqr = ecdf_ub_model.predict(x_train_mean) - ecdf_lb_model.predict(x_train_mean)
y_pred_test_iqr = ecdf_ub_model.predict(x_test_mean) - ecdf_lb_model.predict(x_test_mean)
y_pred_ood_iqr = ecdf_ub_model.predict(x_ood_mean) - ecdf_lb_model.predict(x_ood_mean)

# now calculate F1-SCORES

def binary_pred(x_test, x_threshold):
    return np.piecewise(x_test, [x_test <= x_threshold, x_test > x_threshold], [0, 1])

# x_threshold = 0.5
f1_scores = []
x_thresholds = np.linspace(0.01,0.99,100)

for x_threshold in x_thresholds:
    y_binary_test = binary_pred(y_pred_test_median, x_threshold=x_threshold)
    y_binary_ood = binary_pred(y_pred_ood_median, x_threshold=x_threshold)

    # f1_score_sample = f1_score(np.concatenate((y_binary_test,y_binary_ood)),
    #                            np.concatenate((y_test,y_ood)))
    # f1_scores.append(f1_score_sample)
    f1_score_sample = matthews_corrcoef(np.concatenate((y_binary_test,y_binary_ood)),
                               np.concatenate((y_test,y_ood)))
    # f1_score_sample = f1_score(np.concatenate((y_binary_test,y_binary_ood)),
    #                            np.concatenate((y_test,y_ood)))
    f1_scores.append(f1_score_sample)

iqr = ecdf_ub_model.predict(x_thresholds) - ecdf_lb_model.predict(x_thresholds)

fig, (ax1,ax2) = plt.subplots(2,1)
ax1.plot(x_thresholds, f1_scores)
ax2.scatter(y_pred_train_median, y_pred_train_iqr)



# plt.figure()
# plt.scatter(y_pred_train_median, y_pred_train_iqr)
# plt.scatter(y_pred_ood_median, y_pred_ood_iqr)
#





