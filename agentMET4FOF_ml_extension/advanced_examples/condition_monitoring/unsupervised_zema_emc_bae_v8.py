import copy
import functools
import os
import random
from itertools import combinations

import dill
import numpy as np
import pandas as pd
from captum.attr import DeepLift, GradientShap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import spearmanr, pearsonr

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent, AgentMET4FOF, Coalition
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagateTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent, PRONOSTIA_DatastreamAgent
from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_PlottingAgent, ML_EvaluateAgent, \
    ML_AggregatorAgent, ML_BaseAgent
from agentMET4FOF_ml_extension.ml_experiment import ML_ExperimentLite
from agentMET4FOF_ml_extension.util.calc_drift_rank import calc_gmean_sser, calc_sser_mcc
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.util.helper import move_axis, flatten_dict
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
from progress.bar import Bar

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use(
    'Agg')  # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread


# set drift
# evaluate ood drifts gsser
# evaluate gsdc

def moving_average(dt, window_size=100, mode="mean"):
    pd_temp = pd.DataFrame(dt)
    df_output_smooth = pd_temp.rolling(window=window_size)

    # apply operation depending on mode
    if mode == "mean":
        df_output_smooth = df_output_smooth.mean()
    elif mode == "std":
        df_output_smooth = df_output_smooth.std()
    elif mode == "var":
        df_output_smooth = df_output_smooth.var()

    # return form factor
    if df_output_smooth.values.shape[-1] == 1:
        return df_output_smooth.fillna(method="bfill").values.squeeze(-1)
    else:
        return df_output_smooth.fillna(method="bfill").values

def sensor_moving_average(dt, window_size=100):
    new_dt = dt.copy()
    if len(dt.shape) >= 3:
        for sensor_i in range(dt.shape[1]):
            new_dt[:, sensor_i] = moving_average(dt[:, sensor_i], window_size)
    else:
        new_dt = moving_average(new_dt, window_size=window_size)
    return new_dt

def apply_dict(data_dict, function, **func_params):
    """
    Apply a function on every value of the dictionary

    Returns
    -------
    result : dict
        Dictionary of the same keys, and the function applied on their values
    """
    if isinstance(data_dict, dict):
        return {key: function(val, **func_params) for key, val in data_dict.items()}
    else:
        return function(data_dict, **func_params)


def apply_dict_xy(x_dict, y_dict, function, **func_params):
    """
    Similar to `apply_dict` but takes in two dicts : one of x and one of y

    """
    return {key: function(x_val, y_dict[key], **func_params) for key, x_val in x_dict.items()}


def sum_nll_features_(nll):
    return np.mean(nll, axis=-1)


def sum_nll_features(dict_mean_var):
    return apply_dict(dict_mean_var, np.mean, axis=-1)


def normalise_1_(dt):
    new_dt = dt.copy()
    for sensor_i in range(dt.shape[1]):
        new_dt[:, sensor_i] = dt[:, sensor_i] / dt[0, sensor_i]
    return new_dt


def normalise_1(data_dict):
    return apply_dict(data_dict, function=normalise_1_)


def truncate_first(dt_x, num_samples):
    if isinstance(dt_x, pd.DataFrame):
        dt_x = dt_x.iloc[num_samples:]
    else:
        dt_x = dt_x[num_samples:]
    return dt_x

def mean_bae_samples(dt, apply_mean_last=True):
    if apply_mean_last:
        return {"mean": dt.mean(0).sum(-1)}
    else:
        return {"mean": dt.mean(0), "var": dt.var(0)}

def mean_var_bae_samples(dt, apply_mean_last=True):
    if apply_mean_last:
        return {"mean": dt.mean(0).sum(-1), "var": dt.var(0).sum(-1)}
    else:
        return {"mean": dt.mean(0), "var": dt.var(0)}

def plot_nll_sensors_(nll_mean_var_dict, y_target, metadata=[], figsize=(16, 5)):
    nll_mean = nll_mean_var_dict["mean"]
    if "var" in list(nll_mean_var_dict.keys()):
        var_key_available = True
        nll_var = nll_mean_var_dict["var"]
    else:
        var_key_available = False

    if var_key_available:
        fig, axes = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes = np.array([axes])

    for sensor_i in range(nll_mean.shape[-1]):
        axes[0].plot(y_target.values[:, 0], nll_mean[:, sensor_i])
        if var_key_available:
            axes[1].plot(y_target.values[:, 0], nll_var[:, sensor_i])

    if len(metadata) > 0:
        fig.legend(metadata["labels"])
    fig.tight_layout()

    return fig


def plot_nll_sensors(nll_mean_var_dict, y_target, metadata=[]):
    new_dict = apply_dict_xy(nll_mean_var_dict, y_target, plot_nll_sensors_, metadata=metadata)
    return list(new_dict.values())[0]

def plot_correlation_nll(nll_mean_var_dict, y_target, perturbed_sensors=[], total_sensors=11, figsize=(8,5)):
    key = list(nll_mean_var_dict.keys())[0]
    nll_mean_var = nll_mean_var_dict[key]
    nll_mean = nll_mean_var["mean"]
    if "var" in list(nll_mean_var.keys()):
        has_var = True
        nll_var = nll_mean_var["var"]
    else:
        has_var = False
    y_target = y_target[key].values
    # if y_target[0] > 1:
    #     y_target *= 0.01
    y_target[-1] = 0.95

    # non-perturbed
    unperturbed_sensors = np.array([i for i in range(total_sensors) if i not in perturbed_sensors])
    perturbed_sensors = np.array(perturbed_sensors)
    nll_mean_perturbed = nll_mean[:,perturbed_sensors].mean(-1)
    nll_mean_unperturbed = nll_mean[:,unperturbed_sensors].mean(-1)

    if has_var:
        nll_var_perturbed = nll_var[:,perturbed_sensors].mean(-1)
        nll_var_unperturbed = nll_var[:,unperturbed_sensors].mean(-1)
        pearson_corr_1 = pearsonr(nll_var_perturbed, nll_var_unperturbed)[0]

    pearson_corr_0 = pearsonr(nll_mean_perturbed, nll_mean_unperturbed)[0]

    # actual plotting
    nondrift_color = "tab:orange"
    drift_color = "tab:blue"

    fig, axes = plt.subplots(2,2, figsize=figsize)
    axes= axes.flatten()

    # nll_perturbed
    axes[0].plot(y_target, nll_mean_unperturbed, color=nondrift_color)
    axes_0_twin = axes[0].twinx()
    axes_1_twin = axes[1].twinx()
    axes_0_twin.plot(y_target, nll_mean_perturbed, color=drift_color)

    if has_var:
        axes[1].plot(y_target, nll_var_unperturbed,  color=nondrift_color)
        axes_1_twin.plot(y_target, nll_var_perturbed, color=drift_color)
        axes[3].scatter(nll_var_perturbed, nll_var_unperturbed, c=y_target)

    # bottom plots
    axes[2].scatter(nll_mean_perturbed, nll_mean_unperturbed, c=y_target)

    # set labels symbols
    # nondrift_text = r"$\mathrm{S^{\bot{drift}}}$"
    # drift_text = r"$\mathrm{S^{drift}}$"
    # e_nll_text = r"$\mathrm{E_{\theta}(NLL)}$"
    # var_nll_text = r"$\mathrm{Var_{\theta}(NLL)}$"

    nondrift_text = "Non-Drifting"
    drift_text = "Drifting"
    e_nll_text = r"$\mathrm{E_{\theta}(NLL)}$"
    var_nll_text = r"$\mathrm{Var_{\theta}(NLL)}$"

    # set labels
    axes[0].set_ylabel(e_nll_text+","+nondrift_text, color=nondrift_color)
    axes_0_twin.set_ylabel(e_nll_text+","+drift_text, color=drift_color)
    axes[1].set_ylabel(var_nll_text+","+nondrift_text, color=nondrift_color)
    axes_1_twin.set_ylabel(var_nll_text+","+drift_text, color=drift_color)
    axes[0].tick_params(axis='y', labelcolor=nondrift_color)
    axes_0_twin.tick_params(axis='y', labelcolor=drift_color)
    axes[1].tick_params(axis='y', labelcolor=nondrift_color)
    axes_1_twin.tick_params(axis='y', labelcolor=drift_color)

    # axes[0].set_xticks([0.25, 0.35, 0.45, 0.55,0.65,0.75,0.85,0.95])
    # axes[1].set_xticks([0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
    axes[0].set_xticks([0.25, 0.50, 0.75, 0.95])
    axes[1].set_xticks([0.25, 0.50, 0.75, 0.95])

    axes[0].set_xlabel("Degradation")
    axes[1].set_xlabel("Degradation")

    axes[2].set_xlabel(e_nll_text+","+drift_text)
    axes[2].set_ylabel(e_nll_text+","+nondrift_text)

    axes[3].set_xlabel(var_nll_text+","+drift_text)
    axes[3].set_ylabel(var_nll_text+","+nondrift_text)

    axes[2].text(.5, .9, "Pearson: %.2f" % pearson_corr_0,
            horizontalalignment='center',
            transform=axes[2].transAxes)

    if has_var:
        axes[3].text(.5, .9, "Pearson: %.2f" % pearson_corr_1,
                horizontalalignment='center',
                transform=axes[3].transAxes)

    # set color bar
    norm = plt.Normalize(y_target.min(), y_target.max())
    cmap = plt.get_cmap("viridis")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    axins = inset_axes(axes[3],
                       width="5%",  # width = 5% of parent_bbox width
                       height="98%",  # height : 50%
                       loc='lower left',
                       bbox_to_anchor=(1.05, 0., 1, 1),
                       bbox_transform=axes[3].transAxes,
                       borderpad=0,
                       )

    # cbar = fig.colorbar(sm, ax=ax, pad=0.2)
    cbar = fig.colorbar(sm, cax=axins, ticks=[0.25, 0.50, 0.75, 0.95])

    # cbar = fig.colorbar(sm, cax=axins, ticks=[0.25, 0.5, 0.7, 0.95])

    fig.tight_layout()

    if has_var:
        return fig, {"pearson-mean":pearson_corr_0,
                     "pearson-var":pearson_corr_1}, \
               {"nll-mean-drift":nll_mean_perturbed,
                "nll-mean-nondrift": nll_mean_unperturbed,
                "nll-var-drift": nll_var_perturbed,
                "nll-var-nondrift": nll_var_unperturbed,
                }
    else:
        return fig, {"pearson-mean": pearson_corr_0},{"nll-mean-drift":nll_mean_perturbed,
                "nll-mean-nondrift": nll_mean_unperturbed,
                }

class PlotCorrelationAgent(ML_BaseAgent):
    def init_parameters(self,ml_exp:ML_ExperimentLite, perturbed_sensors = [0], total_sensors = 11, base_folder="MLEXP/"):
        self.perturbed_sensors = perturbed_sensors
        self.total_sensors = total_sensors
        self.ml_exp = ml_exp
        self.init_dmm(use_dmm=True)
        self.base_folder = ml_exp.base_folder

        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

    def on_received_message(self, message):
        if message["channel"] == "test":
            fig, result_dict, trace_dict = plot_correlation_nll(message["data"]["quantities"], message["data"]["target"],
                                       perturbed_sensors=self.perturbed_sensors, total_sensors=self.total_sensors)
            self.save_ml_result(result_dict)
            encoded_name = self.dmm.encode(str(self.ml_exp.ml_parameters),return_pickle=False)
            with open(self.base_folder+encoded_name+".p","wb") as f:
                dill.dump(trace_dict,f)
            fig.savefig(self.base_folder+encoded_name + ".png", dpi=500)
            self.send_plot(fig)

def calc_spearmanr_sensor_np(sensor_nll, y_target: pd.DataFrame, sensor_axis=-1, alpha=0.05):
    spearmanr_scores = [spearmanr(sensor_nll[:, sensor_i], y_target.values[:, 0]) for sensor_i in
                        range(sensor_nll.shape[sensor_axis])]
    spearmanr_scores = [np.abs(sdc.correlation) if sdc.pvalue <= alpha else 0 for sdc in spearmanr_scores]
    spearmanr_scores = np.array(spearmanr_scores)
    return spearmanr_scores

def calc_gmean_sdc(spearman_scores, perturbed_sensors: list):
    total_sensors = len(spearman_scores)
    tpr_spearman_scores = np.mean(
        [spearman_scores[sensor_i] for sensor_i in range(total_sensors) if sensor_i in perturbed_sensors])
    tnr_spearman_scores = np.mean(
        [(1-spearman_scores[sensor_i]) for sensor_i in range(total_sensors) if sensor_i not in perturbed_sensors])
    gmean_sdc = np.sqrt(tpr_spearman_scores * tnr_spearman_scores)

    return {"tpr-spman": tpr_spearman_scores, "tnr-spman": tnr_spearman_scores, "gmean-sdc": gmean_sdc}

def calc_spearmanr_sensor_dict(sensor_nll: dict, y_target: pd.DataFrame, perturbed_sensors: list):
    """
    For dict of keys "mean" and "var"
    """
    spearmanr_scores_dict = apply_dict(sensor_nll,
                                       function=calc_spearmanr_sensor_np,
                                       y_target=y_target)
    gmean_sdc_dict = apply_dict(spearmanr_scores_dict,
                                function=calc_gmean_sdc,
                                perturbed_sensors=perturbed_sensors)

    return gmean_sdc_dict


def calc_spearmanr_sensor(nll_mean_var_dict, y_target, perturbed_sensors):
    """
    Assume the first level keys are the trajectory number.
    Second level key is "mean" "var" of the sensor NLL.

    """
    res = apply_dict_xy(nll_mean_var_dict, y_target, calc_spearmanr_sensor_dict, perturbed_sensors=perturbed_sensors)
    flattened_res = flatten_dict(res)
    return flattened_res

def set_real_drift(x_test, perturb_sensors=[4], min_arg=0, max_arg=100, random_seed=123):
    # IDEA: SET ONLY N SENSORS TO BE ON THE ACTUAL TRAJECTORY
    # THE OTHERS WILL BE RANDOMLY SAMPLED FROM THE "HEALTHY" SET
    np.random.seed(random_seed)

    x_test_ = copy.deepcopy(x_test)
    unperturbed_sensors = np.array([i for i in np.arange(x_test_.shape[1]) if i not in perturb_sensors])

    for unperturbed_sensor_i in unperturbed_sensors:
        random_integers = np.random.randint(min_arg, max_arg, size=x_test_.shape[0])
        x_test_[:, unperturbed_sensor_i] = x_test_[random_integers, unperturbed_sensor_i]
    return x_test_

class ML_DriftAgent(ML_TransformAgent):
    """
    Cycles through the list of perturbed sensors
    """
    def init_parameters(self, perturbed_sensors_full=[], **params):
        super(ML_DriftAgent, self).init_parameters(**params)
        self.perturbed_sensors_full = perturbed_sensors_full
        self.count = 0
        self.perturbed_sensors = self.perturbed_sensors_full[self.count]

    def instantiate_model(self):
        self.perturbed_sensors = self.perturbed_sensors_full[self.count]
        self.model_params.update({"perturbed_sensors":self.perturbed_sensors})
        super(ML_DriftAgent, self).instantiate_model()

    def on_received_message(self, message):
        if message["channel"] == "test":
            self.instantiate_model()
            self.count+=1

def calc_gmean_sser_dict(nll_mean_var_dict, y_target, perturbed_sensors):
    temp_func = functools.partial(apply_dict, function=calc_gmean_sser, perturbed_sensors=perturbed_sensors)
    res = apply_dict(nll_mean_var_dict, function=temp_func)
    return res

def calc_sser_mcc_dict(nll_mean_var_dict, y_target, perturbed_sensors):
    temp_func = functools.partial(apply_dict, function=calc_sser_mcc, perturbed_sensors=perturbed_sensors)
    res = apply_dict(nll_mean_var_dict, function=temp_func)
    return res

def select_sensor_stream(data, select_sensor=0, sensor_axis=1):
    return np.take(data, indices=[select_sensor], axis=sensor_axis)

def form_preproc(agentNetwork, perturbed_sensors, window_size, random_state=3):
    # preprocessing agents
    move_axis_agent = agentNetwork.add_agent(name="MoveXis", agentType=ML_TransformAgent, model=move_axis)

    fft_agent = agentNetwork.add_agent(name="FFT_Agent",
                                       agentType=ML_TransformAgent,
                                       model=FFT_Sensor,
                                       sensor_axis=1
                                       )

    minmax_agent = agentNetwork.add_agent(name="MinMaxScaler",
                                          agentType=ML_TransformAgent,
                                          model=MultiMinMaxScaler,
                                          )

    set_drift_agent = agentNetwork.add_agent(name="SetDrift", agentType=ML_TransformAgent,
                                             model=set_real_drift,
                                             perturb_sensors=perturbed_sensors,
                                             min_arg=0, max_arg=window_size,
                                             random_seed=random_state
                                             )

    # connect agents
    move_axis_agent.bind_output(fft_agent, channel=["train","test","dmm_code"])
    fft_agent.bind_output(minmax_agent, channel=["train","test","dmm_code"])
    minmax_agent.bind_output(set_drift_agent, channel=["test"])

    agentNetwork.add_coalition(name="Preprocessing",agents=[move_axis_agent, fft_agent,minmax_agent,set_drift_agent])
    return move_axis_agent, fft_agent, minmax_agent, set_drift_agent


def connect_preproc(datastream_agent, move_axis_agent, minmax_agent, set_drift_agent, bae_agent):
    """
    Describes how the preprocessing agents are connected to the datastream (input) and to the bae_agent (output).
    """
    # connects datastream to move axis agent
    datastream_agent.bind_output(move_axis_agent, channel=["train", "test", "dmm_code"])


    # connects preprocessing agents to bae (handles both central and coalition)
    minmax_agent.bind_output(set_drift_agent, channel=["test"])

    # connect to coalition bae
    if isinstance(bae_agent, Coalition):
        for agent in bae_agent.agents:
            if "SelectSensor" in agent.name:
                minmax_agent.bind_output(agent, channel=["train", "dmm_code"])
                set_drift_agent.bind_output(agent, channel=["test"])

    # connect to central bae
    else:
        minmax_agent.bind_output(bae_agent, channel=["train", "dmm_code"])
        set_drift_agent.bind_output(bae_agent, channel=["test"])

# #v1
# def form_postproc(agentNetwork, bae_samples, window_size, normalise=True):
#
#     # sum_propagate_agent = agentNetwork.add_agent(name="SumPropagateFeatures", agentType=PropagateTransformAgent,
#     #                                              model=sum_nll_features_, use_dmm=False)
#     sma_agent = agentNetwork.add_agent(name="SMA", agentType=PropagateTransformAgent,
#                                        model=sensor_moving_average,
#                                        window_size=window_size,
#                                        use_dmm=False)
#
#     mean_var_bae_agent = agentNetwork.add_agent(name="MeanVarBAESamples" if bae_samples > 1 else "MeanBAESamples", agentType=ML_TransformAgent,
#                                                 model=mean_var_bae_samples if bae_samples > 1 else mean_bae_samples, use_dmm=False, apply_mean_last=True)
#
#     if normalise:
#         normalise_agent = agentNetwork.add_agent(name="Normalise", agentType=ML_TransformAgent,
#                                                  model=apply_dict,
#                                                  function=normalise_1_,
#                                                  use_dmm=False)
#
#     # truncate the training data from evaluation
#     truncate_agent = agentNetwork.add_agent(name="Truncate", agentType=ML_TransformAgent,
#                                             model=apply_dict,
#                                             function=truncate_first,
#                                             num_samples=window_size,
#                                             func_apply_y=True,  # truncate apply also on Y
#                                             use_dmm=False)
#
#     # bind the agents
#     sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#
#     # sum_propagate_agent.bind_output(sma_agent, channel=["test", "dmm_code"])
#     # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#
#     # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#     # mean_var_bae_agent.bind_output(sum_propagate_agent, channel=["test", "dmm_code"])
#
#     # if normalise:
#     #     sum_propagate_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
#     #     normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#     # else:
#     #     sum_propagate_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#
#     if normalise:
#         mean_var_bae_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
#         normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#     else:
#         mean_var_bae_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#
#     # postproc_agents = agentNetwork.add_coalition(name="Postprocessing", agents=[sum_propagate_agent, sma_agent, mean_var_bae_agent, normalise_agent, truncate_agent])
#     postproc_agents = agentNetwork.add_coalition(name="Postprocessing", agents=[sma_agent, mean_var_bae_agent, normalise_agent, truncate_agent])
#
#     return postproc_agents

#v2
# def form_postproc(agentNetwork, bae_samples, window_size, normalise=True):
#
#     sum_propagate_agent = agentNetwork.add_agent(name="SumPropagateFeatures", agentType=PropagateTransformAgent,
#                                                  model=sum_nll_features_, use_dmm=False)
#     mean_var_bae_agent = agentNetwork.add_agent(name="MeanVarBAESamples" if bae_samples > 1 else "MeanBAESamples", agentType=ML_TransformAgent,
#                                                 model=mean_var_bae_samples if bae_samples > 1 else mean_bae_samples, use_dmm=False, apply_mean_last=False)
#
#     sma_agent = agentNetwork.add_agent(name="SMA", agentType=PropagateTransformAgent,
#                                        model=sensor_moving_average,
#                                        window_size=window_size,
#                                        use_dmm=False)
#
#     if normalise:
#         normalise_agent = agentNetwork.add_agent(name="Normalise", agentType=ML_TransformAgent,
#                                                  model=apply_dict,
#                                                  function=normalise_1_,
#                                                  use_dmm=False)
#
#     # truncate the training data from evaluation
#     truncate_agent = agentNetwork.add_agent(name="Truncate", agentType=ML_TransformAgent,
#                                             model=apply_dict,
#                                             function=truncate_first,
#                                             num_samples=window_size,
#                                             func_apply_y=True,  # truncate apply also on Y
#                                             use_dmm=False)
#
#     # bind the agents
#     sum_propagate_agent.bind_output(sma_agent, channel=["test", "dmm_code"])
#     sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#
#
#     # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#
#     # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
#     # mean_var_bae_agent.bind_output(sum_propagate_agent, channel=["test", "dmm_code"])
#
#     # if normalise:
#     #     sum_propagate_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
#     #     normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#     # else:
#     #     sum_propagate_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#
#     if normalise:
#         mean_var_bae_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
#         normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#     else:
#         mean_var_bae_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
#
#     postproc_agents = agentNetwork.add_coalition(name="Postprocessing", agents=[sum_propagate_agent, sma_agent, mean_var_bae_agent, normalise_agent, truncate_agent])
#     # postproc_agents = agentNetwork.add_coalition(name="Postprocessing", agents=[mean_var_bae_agent,sma_agent, normalise_agent, truncate_agent])
#
#     return postproc_agents


# v3
def form_postproc(agentNetwork, bae_samples, window_size, normalise=True):

    # sum_propagate_agent = agentNetwork.add_agent(name="SumPropagateFeatures", agentType=PropagateTransformAgent,
    #                                              model=sum_nll_features_, use_dmm=False)
    mean_var_bae_agent = agentNetwork.add_agent(name="MeanVarBAESamples" if bae_samples > 1 else "MeanBAESamples", agentType=ML_TransformAgent,
                                                model=mean_var_bae_samples if bae_samples > 1 else mean_bae_samples, use_dmm=False)

    sma_agent = agentNetwork.add_agent(name="SMA", agentType=ML_TransformAgent,
                                       model=apply_dict,
                                       function=sensor_moving_average,
                                       window_size=window_size,
                                       use_dmm=False)

    if normalise:
        normalise_agent = agentNetwork.add_agent(name="Normalise", agentType=ML_TransformAgent,
                                                 model=apply_dict,
                                                 function=normalise_1_,
                                                 use_dmm=False)

    # truncate the training data from evaluation
    truncate_agent = agentNetwork.add_agent(name="Truncate", agentType=ML_TransformAgent,
                                            model=apply_dict,
                                            function=truncate_first,
                                            num_samples=window_size,
                                            func_apply_y=True,  # truncate apply also on Y
                                            use_dmm=False)

    # bind the agents
    mean_var_bae_agent.bind_output(sma_agent, channel=["test", "dmm_code"])

    # sum_propagate_agent.bind_output(sma_agent, channel=["test", "dmm_code"])
    # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])

    # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
    # mean_var_bae_agent.bind_output(sum_propagate_agent, channel=["test", "dmm_code"])

    # if normalise:
    #     sum_propagate_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
    #     normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
    # else:
    #     sum_propagate_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])

    if normalise:
        sma_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
        normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
        postproc_agents = agentNetwork.add_coalition(name="Postprocessing",
                                                     agents=[mean_var_bae_agent, sma_agent, normalise_agent,
                                                             truncate_agent])

    else:
        sma_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])
        postproc_agents = agentNetwork.add_coalition(name="Postprocessing",
                                                     agents=[mean_var_bae_agent, sma_agent, truncate_agent])

    return postproc_agents


def form_central_bae(agentNetwork, bae_params):
    """
    Forms the agents which make up the central BAE postprocessings
    """
    bae_samples = bae_params["bae_samples"]
    cbae_agent = agentNetwork.add_agent(name="Central_BAE_Agent" if bae_samples > 1 else "Central_AE_Agent", agentType=CBAE_Agent,
                                        **bae_params
                                        )


    return cbae_agent

def form_coalition_bae(agentNetwork, bae_params, total_sensors):
    """
    Forms the agents which make up the central BAE postprocessings
    """

    # setup coalition bae
    cbae_agents = []
    select_sensor_agents = []
    aggregator = agentNetwork.add_agent(name="Aggregator",
                                        agentType=ML_AggregatorAgent,
                                        concat_axis = 2
                                        )

    for sensor_i in range(total_sensors):
        select_sensor_agent = agentNetwork.add_agent(name="SelectSensor_"+str(sensor_i+1),
                                               agentType=ML_TransformAgent,
                                               model=select_sensor_stream,
                                               select_sensor=sensor_i)
        conv_architecture = bae_params["conv_architecture"]
        conv_architecture[0] = 1
        bae_params.update({"conv_architecture": conv_architecture})
        cbae_agent = agentNetwork.add_agent(name="Coalition_BAE_Agent"+str(sensor_i+1), agentType=CBAE_Agent,
                                            **bae_params
                                            )

        select_sensor_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])
        cbae_agent.bind_output(aggregator, channel=["test"])

        # add to list of agents
        select_sensor_agents.append(select_sensor_agent)
        cbae_agents.append(cbae_agent)

    coalition_bae = agentNetwork.add_coalition("CoalitionBAE", agents=select_sensor_agents+cbae_agents)

    return coalition_bae, aggregator

def get_combination_perturbed(total_sensors=11, samples=-1, random_seed=123):
    random.seed(random_seed)

    if samples<=0:
        comb_perturbed = [list(combinations(np.arange(total_sensors), i)) for i in np.arange(1,total_sensors)]
    else:
        temp_combs = [list(combinations(np.arange(total_sensors), i)) for i in np.arange(1,total_sensors)]
        comb_perturbed = [list(random.sample(temp_comb,samples)) if samples < len(temp_comb) else temp_comb for temp_comb in temp_combs]
    return comb_perturbed

class OrchestratorAgent(AgentMET4FOF):
    def init_parameters(self, agentNetwork, total_sensors:int,ml_exp:ML_ExperimentLite, window_size, random_state,
                        spman_eval_agent,
                        rank_eval_agent,
                        plot_correlation_agent,
                        set_drift_agent,
                        ):

        self.agentNetwork = agentNetwork
        # self.bae_agent = bae_agent
        self.count = 0
        self.new_count = 0
        # self.bae_params = bae_params
        all_combinations = get_combination_perturbed(total_sensors=total_sensors)

        if total_sensors>2:
            self.combinations = all_combinations[0] + all_combinations[-1]
        else:
            self.combinations = all_combinations[-1]

        # self.combinations = self.combinations[-1]
        # self.combinations = [all_combinations[-1][-2],all_combinations[-1][-1]]
        self.ml_exp = ml_exp


        self.window_size = window_size
        self.random_state = random_state

        self.spman_eval_agent = spman_eval_agent
        self.rank_eval_agent = rank_eval_agent
        self.plot_correlation_agent = plot_correlation_agent
        self.set_drift_agent = set_drift_agent

    def any_agent_has_message(self):
        agents = self.agentNetwork.agents()
        for agent in agents:
            if len(self.agentNetwork.get_agent(agent).mesa_message_queue) >0:
                return True
        return False

    def update_eval_agents(self):
        perturbed_sensors = list(self.combinations[self.count])
        # perturbed_sensors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        eval_spman = functools.partial(calc_spearmanr_sensor, perturbed_sensors=perturbed_sensors)
        eval_rank = functools.partial(calc_sser_mcc_dict, perturbed_sensors=perturbed_sensors)

        self.spman_eval_agent.forward_model = eval_spman
        self.rank_eval_agent.forward_model = eval_rank

        #
        # self.spman_eval_agent.model_params.update({"perturbed_sensors":perturbed_sensors})
        # self.rank_eval_agent.model_params.update({"perturbed_sensors": perturbed_sensors})
        # self.spman_eval_agent.instantiate_model()
        # self.rank_eval_agent.instantiate_model()

    def set_sensor_drift(self):
        perturbed_sensors = list(self.combinations[self.count])
        # perturbed_sensors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        wrapped_set_real_drift = functools.partial(set_real_drift,
                                                   perturb_sensors = perturbed_sensors,
                                                   min_arg = 0,
                                                   max_arg = self.window_size, random_seed = self.random_state)
        self.ml_exp.ml_parameters.update({"perturbed": perturbed_sensors,
                                          "n_perturbed": len(perturbed_sensors)})
        # self.send_set_attr(attr="forward_model", value=wrapped_set_real_drift)  # send to set drift agent
        self.set_drift_agent.forward_model = wrapped_set_real_drift
        self.plot_correlation_agent.perturbed_sensors=perturbed_sensors

    def agent_loop(self):
        if self.count != self.new_count and (self.count < (len(self.combinations))):
            # make sure all agents have cleared of their message queues (i.e no more actions left)
            if not self.any_agent_has_message():
                self.count = self.new_count
                # self.set_sensor_drift()
                # self.update_eval_agents()

                # first time
                # if self.count == -1:
                #     self.send_request_method("send_train_data")
                self.send_request_method("send_test_data")


    def on_received_message(self, message):
        # defines the inner loop of setting drift sensors
        if self.count < (len(self.combinations)-1) and (self.count == self.new_count):
            self.new_count = self.count+1
        else:
            self.log_info("COMPLETE ALL ITERATIONS")


            # for agent in self.agentNetwork.agents():
            #     self.agentNetwork.get_agent(agent).shutdown()
            # self.agentNetwork.del_coalition()
            # self.agentNetwork.count += 1
            # if self.agentNetwork.count >= len(self.agentNetwork.parameters_list):
            #     exit()
            # main(self.agentNetwork, **self.agentNetwork.parameters_list[self.agentNetwork.count])

def main(agentNetwork=None,
         model_capacity=2,
         axis="2_1",
         dataset="PRONOSTIA",
         n_layers = 2,
         bae_config="central",
         bae_or_ae="bae",
         explanation="nll",
         random_state=8888
         ):
    # start agent network server
    if agentNetwork is None:
        agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # random_state = 4
    # random_state = 3333
    # perturbed_sensors = [5,4,3,2,1]
    perturbed_sensors = [0]

    experiment_parameters = {"perturbed": perturbed_sensors,
                             "n_perturbed": len(perturbed_sensors),
                             "axis": axis,
                             "random_state": random_state,
                             "dataset": dataset,  # {"ZEMA", "PRONOSTIA"}
                             "model_capacity": model_capacity,
                             "n_layers": n_layers,
                             "bae_config" : bae_config, # {"central","coalition"}
                             "bae_or_ae" : bae_or_ae, # {"bae", "ae"}
                             "explanation": explanation,  # {"nll", "gradshap", "deeplift"}
                             "use_lr_finder": False,
                             }  # this will be logged in the csv

    experiment_parameters.update({"model_type":experiment_parameters["bae_or_ae"]+
                                               "-"+
                                               experiment_parameters["bae_config"]+"-"+
                                               experiment_parameters["explanation"]
                                  })
    posthoc_classes = {"gradshap":GradientShap, "deeplift":DeepLift, "nll":None}
    posthoc_class = posthoc_classes[experiment_parameters["explanation"]]

    datastream_params =  {
                         "ZEMA": {"agentType": ZEMA_DatastreamAgent},
                         "PRONOSTIA": {"agentType": PRONOSTIA_DatastreamAgent}
                         }

    ml_exp = ML_ExperimentLite(ml_exp_name="unsupervised-BAE",
                               ml_parameters=experiment_parameters,
                               save_csv=True, base_folder="MLEXP-Explainability/")


    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(
        name=experiment_parameters["dataset"] + "_" + str(experiment_parameters["axis"]),
        agentType=datastream_params[experiment_parameters["dataset"]]["agentType"],
        train_axis=[experiment_parameters["axis"]],
        test_axis=[experiment_parameters["axis"]],
        train_size=0.125,
        random_state=random_state,
        shuffle=False,
        return_full_test=True,
        use_dmm=False,
        cut_first_perc=0.15,
        cut_last_perc=0.05
    )

    # censor first n % and last k %
    window_size = datastream_agent.x_train.shape[0]
    total_sensors = datastream_agent.x_train.shape[-1]
    metadata = datastream_agent.metadata

    bae_params = {
        "conv_architecture": [total_sensors]+ [int(i*experiment_parameters["model_capacity"]) for i in [8, 5, 3][:experiment_parameters["n_layers"]]],
        "dense_architecture": [int(100*experiment_parameters["model_capacity"])],
        # "dense_architecture": [],
        "conv_kernel": [int(i*experiment_parameters["model_capacity"]) for i in [20, 10, 5][:experiment_parameters["n_layers"]]],
        "conv_stride": [4, 2, 1][:experiment_parameters["n_layers"]],
        "latent_dim": int(20*experiment_parameters["model_capacity"]),
        "weight_decay": 0.001,
        "likelihood": "1_gaussian",
        "learning_rate": 0.01,
        "bae_samples": 1 if experiment_parameters["bae_or_ae"] == "ae" else 5,
        "random_state": random_state,
        "use_cuda": True,
        "train_model": True,
        "num_epochs": 250,
        "move_axis": False,
        "use_dmm": True,
        "dmm_samples": 1,
        "return_samples": True,
        "predict_train": False,
        "posthoc_class": posthoc_class,  # {GradientShap, DeepLift}
        "use_lr_finder": experiment_parameters["use_lr_finder"]

    }

    # postprocessing agents
    move_axis_agent, fft_agent, minmax_agent, set_drift_agent = form_preproc(agentNetwork=agentNetwork,
                                                                             perturbed_sensors=perturbed_sensors,
                                                                             window_size=window_size,
                                                                             random_state=random_state)

    postproc_agents = form_postproc(agentNetwork=agentNetwork,
                                    bae_samples=bae_params["bae_samples"],
                                    window_size=window_size,
                                    normalise=True if experiment_parameters["explanation"] == "nll" else False
                                    )

    truncate_agent = postproc_agents.agents[-1]

    if experiment_parameters["bae_config"] == "central":
        cbae_agent = form_central_bae(agentNetwork,bae_params)
        connect_preproc(datastream_agent, move_axis_agent, minmax_agent, set_drift_agent, cbae_agent)
        cbae_agent.bind_output(postproc_agents.agents[0], channel="test")
    elif experiment_parameters["bae_config"] == "coalition":
        cbae_coalition, aggregator_agent = form_coalition_bae(agentNetwork,bae_params,total_sensors)
        aggregator_agent.bind_output(postproc_agents.agents[0],channel="test")
        connect_preproc(datastream_agent, move_axis_agent, minmax_agent, set_drift_agent, cbae_coalition)

    # evaluation agents
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    plotting_agent = agentNetwork.add_agent("PlottingAgent", agentType=ML_PlottingAgent,
                                            model=plot_nll_sensors,
                                            use_dmm=False,
                                            metadata=metadata
                                            )

    plot_correlation_agent = agentNetwork.add_agent("PlottingCorrAgent", agentType=PlotCorrelationAgent,
                                                    perturbed_sensors=perturbed_sensors,
                                                    total_sensors=total_sensors,
                                                    ml_exp = ml_exp,
                                                    )

    evaluate_spearmanr_agent = agentNetwork.add_agent("SpearmanAgent",
                                                      agentType=ML_EvaluateAgent,
                                                      model=calc_spearmanr_sensor,
                                                      perturbed_sensors=perturbed_sensors,
                                                      ml_exp=ml_exp
                                                      )

    evaluate_ranking_agent = agentNetwork.add_agent("RankingAgent",
                                                    agentType=ML_EvaluateAgent,
                                                    model=calc_sser_mcc_dict,
                                                    perturbed_sensors=perturbed_sensors,
                                                    ml_exp=ml_exp
                                                    )

    orchestrator_agent = agentNetwork.add_agent(agentType=OrchestratorAgent,
                                                agentNetwork=agentNetwork,
                                                total_sensors = total_sensors,
                                                ml_exp = ml_exp,
                                                window_size=window_size,
                                                random_state=random_state,
                                                spman_eval_agent = evaluate_spearmanr_agent,
                                                rank_eval_agent=evaluate_ranking_agent,
                                                plot_correlation_agent= plot_correlation_agent,
                                                set_drift_agent=set_drift_agent
                                                )

    truncate_agent.bind_output(evaluate_spearmanr_agent, channel=["test", "dmm_code"])
    truncate_agent.bind_output(evaluate_ranking_agent, channel=["test", "dmm_code"])

    # connect to plotting agents
    truncate_agent.bind_output(plotting_agent, channel=["test"])
    truncate_agent.bind_output(plot_correlation_agent, channel=["test"])
    # connect evaluate agent
    # normalise_agent.bind_output(plotting_agent, channel=["test"])


    plot_correlation_agent.bind_output(monitor_agent, channel=["plot"])
    plotting_agent.bind_output(monitor_agent, channel=["plot"])

    evaluate_spearmanr_agent.bind_output(monitor_agent)
    evaluate_ranking_agent.bind_output(monitor_agent)

    evaluate_ranking_agent.bind_output(orchestrator_agent)
    # orchestrator_agent.bind_output(set_drift_agent, channel="set-attr")
    orchestrator_agent.bind_output(datastream_agent, channel="request-method")

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':

    # dataset_choices = ["PRONOSTIA","ZEMA"]
    dataset_choices = ["PRONOSTIA"]
    axes = {
        # "PRONOSTIA":[
    #     "1_1","1_2","1_3","1_4","1_5","1_6","1_7",
    #         "2_1", "2_2", "2_3", "2_4", "2_5", "2_6",
    #      "2_7",
    #         "3_1", "3_2", "3_3"],
    #         "ZEMA": [3,5,7],
    #     "ZEMA": [7]
        "PRONOSTIA": ["3_1"]
            }

    model_caps = [
                  {"model_capacity":0.5,
                   "n_layers":1,
                   },
                #   {"model_capacity": 0.5,
                #    "n_layers": 2,
                #    },
                #   {"model_capacity": 0.5,
                #    "n_layers": 3,
                #    },
                # {"model_capacity": 1,
                #  "n_layers": 1,
                #  },
                # {"model_capacity": 1,
                #  "n_layers": 2,
                #  },
                # {"model_capacity": 1,
                #  "n_layers": 3,
                #  },
                # {"model_capacity": 2,
                #  "n_layers": 1,
                #  },
                # {"model_capacity": 2,
                #  "n_layers": 2,
                #  },
                # {"model_capacity": 2,
                #  "n_layers": 3,
                #  },
                  ]
    bae_models = [
        # {
        #            "bae_config":"central",
        #            "bae_or_ae":"bae",
        #            "explanation": "nll"
        #            },
        {
            "bae_config": "coalition",
            "bae_or_ae": "bae",
            "explanation": "nll"
        },
        # {
        #     "bae_config": "central",
        #     "bae_or_ae": "ae",
        #     "explanation": "nll"
        # },
        # {
        #     "bae_config": "coalition",
        #     "bae_or_ae": "ae",
        #     "explanation": "nll"
        # },
        # {
        #     "bae_config": "central",
        #     "bae_or_ae": "ae",
        #     "explanation": "gradshap"
        # },
        # {
        #     "bae_config": "central",
        #     "bae_or_ae": "ae",
        #     "explanation": "deeplift"
        # },
    ]

    parameters_combinations =[]
    new_param = {}
    for bae_model in bae_models:
        for model_cap in model_caps:
            for dataset in dataset_choices:
                for axis in axes[dataset]:
                    new_param.update(model_cap)
                    new_param.update(bae_model)
                    new_param.update({"dataset":dataset})
                    new_param.update({"axis": axis})
                    parameters_combinations.append(new_param.copy())
    print(parameters_combinations)

    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")
    agentNetwork.parameters_list = parameters_combinations
    agentNetwork.count = 0
    agentNetwork = main(agentNetwork, **agentNetwork.parameters_list[agentNetwork.count])
