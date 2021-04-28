import functools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import copy

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_PlottingAgent, ML_EvaluateAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagateTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent, PRONOSTIA_DatastreamAgent
from agentMET4FOF_ml_extension.ml_experiment import ML_ExperimentLite
from agentMET4FOF_ml_extension.util.calc_drift_rank import calc_gmean_sser, calc_mcc
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateRegressionClassificationAgent
from agentMET4FOF_ml_extension.util.helper import move_axis, flatten_dict
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
import collections

np.random.seed(100)

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
    for sensor_i in range(dt.shape[1]):
        new_dt[:, sensor_i] = moving_average(dt[:, sensor_i], window_size)
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


def mean_var_bae_samples(dt):
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
    tnr_spearman_scores = 1 - np.mean(
        [spearman_scores[sensor_i] for sensor_i in range(total_sensors) if sensor_i not in perturbed_sensors])
    gmean_sdc = np.sqrt(tpr_spearman_scores * tnr_spearman_scores)

    return {"tpr-spman": tpr_spearman_scores, "tnr-spman": tnr_spearman_scores, "gmean_sdc": gmean_sdc}


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


def calc_gmean_sser_dict(nll_mean_var_dict, y_target, perturbed_sensors):
    temp_func = functools.partial(apply_dict, function=calc_gmean_sser, perturbed_sensors=perturbed_sensors)
    res = apply_dict(nll_mean_var_dict, function=temp_func)
    return res

def select_sensor_stream(data, select_sensor=0, sensor_axis=1):
    return np.take(data,indices=[select_sensor], axis=sensor_axis)


def main():
    random_state = 4
    perturbed_sensors = [0]
    experiment_parameters = {"perturbed": perturbed_sensors,
                             "axis": "2_1",
                             "random_state": random_state,
                             "dataset": "PRONOSTIA",  # {"ZEMA", "PRONOSTIA"}
                             "model_capacity": 4,
                             "n_layers": 3,
                             "model_type": "bae-central",
                             # {"bae-central", "bae-coalition", "ae-central", "ae-coalition"}
                             "explanation": "nll"  # {"nll", "gradshap", "ig"}
                             }  # this will be logged in the csv

    datastream_params = {"ZEMA": {"agentType": ZEMA_DatastreamAgent},
                         "PRONOSTIA": {"agentType": PRONOSTIA_DatastreamAgent}
                         }

    ml_exp = ML_ExperimentLite(ml_exp_name="unsupervised-BAE",
                               ml_parameters=experiment_parameters,
                               save_csv=True)

    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(
        name=experiment_parameters["dataset"] + "_" + experiment_parameters["axis"],
        agentType=datastream_params[experiment_parameters["dataset"]]["agentType"],
        train_axis=[experiment_parameters["axis"]],
        test_axis=[experiment_parameters["axis"]],
        train_size=0.1875,
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

    cbae_agent = agentNetwork.add_agent(name="CBAE_Agent", agentType=CBAE_Agent,
                                        conv_architecture=[total_sensors, 8, 3],
                                        dense_architecture=[100],
                                        conv_kernel=[100, 10],
                                        conv_stride=[10, 4],
                                        latent_dim=10,
                                        weight_decay=0.015,
                                        likelihood="1_gaussian",
                                        learning_rate=0.01,
                                        bae_samples=5,
                                        random_state=random_state,
                                        use_cuda=True,
                                        train_model=True,
                                        num_epochs=5,
                                        move_axis=False,
                                        use_dmm=True,
                                        dmm_samples=1,
                                        return_samples=True,
                                        predict_train=False
                                        )



    sum_propagate_agent = agentNetwork.add_agent(name="SumPropagateFeatures", agentType=PropagateTransformAgent,
                                                 model=sum_nll_features_, use_dmm=False)
    sma_agent = agentNetwork.add_agent(name="SMA", agentType=PropagateTransformAgent,
                                       model=sensor_moving_average,
                                       window_size=window_size,
                                       use_dmm=False)

    mean_var_bae_agent = agentNetwork.add_agent(name="MeanVarBAESamples", agentType=ML_TransformAgent,
                                                model=mean_var_bae_samples, use_dmm=False)

    sum_agent = agentNetwork.add_agent(name="SumFeatures", agentType=ML_TransformAgent,
                                       model=sum_nll_features, use_dmm=False)

    normalise_agent = agentNetwork.add_agent(name="Normalise", agentType=ML_TransformAgent,
                                             model=apply_dict,
                                             function=normalise_1_,
                                             use_dmm=False)

    truncate_agent = agentNetwork.add_agent(name="Truncate", agentType=ML_TransformAgent,
                                            model=apply_dict,
                                            function=truncate_first,
                                            num_samples=window_size,
                                            func_apply_y=True,  # truncate apply also on Y
                                            use_dmm=False)

    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    plotting_agent = agentNetwork.add_agent("PlottingAgent", agentType=ML_PlottingAgent,
                                            model=plot_nll_sensors,
                                            use_dmm=False,
                                            metadata=metadata
                                            )

    evaluate_spearmanr_agent = agentNetwork.add_agent("SpearmanAgent",
                                                      agentType=ML_EvaluateAgent,
                                                      model=calc_spearmanr_sensor,
                                                      perturbed_sensors=perturbed_sensors,
                                                      ml_exp=ml_exp
                                                      )

    evaluate_ranking_agent = agentNetwork.add_agent("RankingAgent",
                                                    agentType=ML_EvaluateAgent,
                                                    model=calc_gmean_sser_dict,
                                                    perturbed_sensors=perturbed_sensors,
                                                    ml_exp=ml_exp
                                                    )

    # connect fft agent to cbae agent
    datastream_agent.bind_output(move_axis_agent, channel=["train", "test", "dmm_code"])
    # datastream_agent.bind_output(plotting_agent, channel="metadata")

    move_axis_agent.bind_output(fft_agent, channel=["train", "test", "dmm_code"])
    fft_agent.bind_output(minmax_agent, channel=["train", "test", "dmm_code"])
    # minmax_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])
    minmax_agent.bind_output(cbae_agent, channel=["train", "dmm_code"])

    minmax_agent.bind_output(set_drift_agent, channel=["test", "dmm_code"])
    set_drift_agent.bind_output(cbae_agent, channel=["test", "dmm_code"])

    # postprocess cbae outputs v1
    cbae_agent.bind_output(sum_propagate_agent, channel=["test", "dmm_code"])
    sum_propagate_agent.bind_output(sma_agent, channel=["test", "dmm_code"])
    sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
    mean_var_bae_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
    normalise_agent.bind_output(truncate_agent, channel=["test", "dmm_code"])

    # postprocess cbae outputs v2
    # cbae_agent.bind_output(sma_agent, channel=["test","dmm_code"])
    # sma_agent.bind_output(mean_var_bae_agent, channel=["test","dmm_code"])
    # mean_var_bae_agent.bind_output(sum_agent, channel=["test","dmm_code"])
    # # sum_agent.bind_output(truncate_agent, channel=["test","dmm_code"])
    # sum_agent.bind_output(normalise_agent, channel=["test", "dmm_code"])
    # normalise_agent.bind_output(truncate_agent, channel=["test","dmm_code"])

    # postprocess cbae outputs v3
    # cbae_agent.bind_output(sma_agent, channel=["test","dmm_code"])
    # sma_agent.bind_output(mean_var_bae_agent, channel=["test", "dmm_code"])
    # mean_var_bae_agent.bind_output(sum_agent, channel=["test","dmm_code"])
    # sum_agent.bind_output(normalise_agent, channel=["test","dmm_code"])
    # normalise_agent.bind_output(truncate_agent, channel=["test","dmm_code"])

    truncate_agent.bind_output(evaluate_spearmanr_agent, channel=["test", "dmm_code"])
    truncate_agent.bind_output(evaluate_ranking_agent, channel=["test", "dmm_code"])

    # connect to plotting agents
    truncate_agent.bind_output(plotting_agent, channel=["test"])

    # connect evaluate agent
    # normalise_agent.bind_output(plotting_agent, channel=["test"])

    plotting_agent.bind_output(monitor_agent, channel=["plot"])
    evaluate_spearmanr_agent.bind_output(monitor_agent)
    evaluate_ranking_agent.bind_output(monitor_agent)

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
