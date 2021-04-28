import os
import pickle

import numpy as np
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

class SaveBAESamples(ML_BaseAgent):
    def on_received_message(self,data):
        print("SAVED SAMPLES!")
        pickle.dump(data,open("raw_bae_nll.p","wb"))

def main():
    random_state = 12
    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=Liveline_DatastreamAgent,
                                              input_stage=1,
                                              target_stage=1,
                                              train_size=0.5,
                                              random_state=random_state,)

    # already has one internally Liveline ds
    # minmax_agent = agentNetwork.add_agent(name="MinMaxScaler",
    #                                       agentType=ML_TransformAgent,
    #                                       model=MultiMinMaxScaler,
    #                                       )

    print(datastream_agent.x_train.shape[-1])
    cbae_agent = agentNetwork.add_agent(name="CBAE_Agent", agentType=CBAE_Agent,
                                        conv_architecture=[],
                                        dense_architecture=[datastream_agent.x_train.shape[-1], 500],
                                        latent_dim=50,
                                        likelihood="homo_gaussian",
                                        learning_rate=0.01,
                                        bae_samples=5,
                                        random_state=random_state,
                                        use_cuda=True,
                                        train_model=True,
                                        num_epochs=3,
                                        move_axis=False,
                                        use_dmm=True,
                                        dmm_samples=1,
                                        return_samples = True
                                        )
    save_inp_agent = agentNetwork.add_agent(agentType=SaveBAESamples)

    move_axis_agent = agentNetwork.add_agent(name="MoveXis", agentType=ML_TransformAgent, model=move_axis)
    mean_var_agent = agentNetwork.add_agent(name="MeanVar", agentType=ML_TransformAgent, model=compute_mean_var, return_dict=True)

    ood_evaluate_agent = agentNetwork.add_agent(agentType=EvaluateAUROCAgent, save_pickle=True)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # connect fft agent to cbae agent
    datastream_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])
    # move_axis_agent.bind_output(minmax_agent, channel=["train","test","dmm_code"])
    # minmax_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])
    cbae_agent.bind_output(save_inp_agent, channel=["test"])

    # connect evaluate agent to cbae
    cbae_agent.bind_output(mean_var_agent, channel=["test"])
    mean_var_agent.bind_output(ood_evaluate_agent, channel=["test"])
    ood_evaluate_agent.bind_output(monitor_agent, channel=["test","plot"])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()

# matplotlib.use('Qt5Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread
#
# anom_threshold = AnomalyThreshold()
#
# x_data = np.random.random_sample(1000)
# anom_threshold.fit(x_train=x_data)
# y_pred = anom_threshold.transform(x_data)
# print(anom_threshold.transform(x_data))
#
# ecdf_model, x_range, ecdf_vals =  anom_threshold.fit_ecdf(x_data)
#
# fig, ax = plt.subplots(1,1)
# anom_threshold.plot_ecdf( x_range, ecdf_vals, ax=ax, complementary=False)
# ax.vlines(x=anom_threshold.x_threshold, ymin=0,ymax=1)
# plt.show()
#
# # counts
# y_zeros = len(np.argwhere(y_pred == 0))
# y_ones = len(np.argwhere(y_pred == 1))
#

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




# anom_threshold = AnomalyThreshold()
#
# x_data = np.random.random_sample(1000)
# anom_threshold.fit(x_train=x_data)
# y_pred = anom_threshold.transform(x_data)
# print(anom_threshold.transform(x_data))
#
# ecdf_model, x_range, ecdf_vals =  anom_threshold.fit_ecdf(x_data)
#
# fig, ax = plt.subplots(1,1)
# anom_threshold.plot_ecdf( x_range, ecdf_vals, ax=ax, complementary=False)
# ax.vlines(x=anom_threshold.x_threshold, ymin=0,ymax=1)
# plt.show()
#
# # counts
# y_zeros = len(np.argwhere(y_pred == 0))
# y_ones = len(np.argwhere(y_pred == 1))













