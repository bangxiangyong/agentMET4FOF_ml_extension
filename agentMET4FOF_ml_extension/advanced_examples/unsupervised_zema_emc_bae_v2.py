import numpy as np
import pandas as pd

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_PlottingAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagateTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateAgent
from agentMET4FOF_ml_extension.util.helper import move_axis
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

np.random.seed(100)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

# set drift
# moving average
# normalise
# evaluate ood

def moving_average(dt, window_size=100, mode="mean"):
    pd_temp = pd.DataFrame(dt)
    df_output_smooth = pd_temp.rolling(window=window_size)

    #apply operation depending on mode
    if mode == "mean":
        df_output_smooth = df_output_smooth.mean()
    elif mode == "std":
        df_output_smooth = df_output_smooth.std()
    elif mode == "var":
        df_output_smooth = df_output_smooth.var()

    #return form factor
    if df_output_smooth.values.shape[-1] == 1:
        return df_output_smooth.fillna(method="bfill").values.squeeze(-1)
    else:
        return df_output_smooth.fillna(method="bfill").values

def sensor_moving_average(dt, window_size=100):
    new_dt = dt.copy()
    for sensor_i in range(dt.shape[1]):
        new_dt[:, sensor_i] = moving_average(dt[:, sensor_i],window_size)
    return new_dt

def apply_dict(data_dict,function, **func_params):
    return {key:function(val, **func_params) for key,val in data_dict.items()}

def apply_dict_xy(x_dict,y_dict,function, **func_params):
    return {key:function(x_val,y_dict[key], **func_params) for key,x_val in x_dict.items()}

def sum_nll_features_(nll):
    return np.mean(nll,axis=-1)

def sum_nll_features(dict_mean_var):
    return apply_dict(dict_mean_var, np.mean,axis=-1)

def normalise_1_(dt):
    new_dt = dt.copy()
    for sensor_i in range(dt.shape[1]):
        new_dt[:, sensor_i] = dt[:,sensor_i]/dt[0,sensor_i]
    return new_dt

def normalise_1(data_dict):
    return apply_dict(data_dict,normalise_1_)

def mean_var_bae_samples(dt):
    return {"mean":dt.mean(0), "var":dt.var(0)}

def plot_nll_sensors_(nll_mean_var_dict, y_target, metadata=[]):

    nll_mean = nll_mean_var_dict["mean"]
    if "var" in list(nll_mean_var_dict.keys()):
        var_key_available = True
        nll_var = nll_mean_var_dict["var"]
    else:
        var_key_available = False

    if var_key_available:
        fig, axes = plt.subplots(1,2)
    else:
        fig, axes = plt.subplots(1,1)
        axes = np.array([axes])

    for sensor_i in range(nll_mean.shape[-1]):
        axes[0].plot(y_target.values[:, 0], nll_mean[:, sensor_i])
        if var_key_available:
            axes[1].plot(y_target.values[:, 0], nll_var[:, sensor_i])

        # plt.plot(nll_mean[:, sensor_i])
        # plt.plot(np.random.random(100))
    # fig.legend(metadata["labels"])
    return fig

def plot_nll_sensors(nll_mean_var_dict, y_target, metadata=[]):
    new_dict = apply_dict_xy(nll_mean_var_dict, y_target, plot_nll_sensors_, metadata=metadata)
    return list(new_dict.values())[0]

def main():
    random_state = 123
    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              id_axis=[5],
                                              test_axis=[5],
                                              train_size=0.2,
                                              random_state=random_state,
                                              shuffle=False,
                                              return_full_test = True,
                                              use_dmm=False
                                              )

    move_axis_agent = agentNetwork.add_agent(name="MoveXis",agentType=ML_TransformAgent, model=move_axis)

    fft_agent = agentNetwork.add_agent(name="FFT_Agent",
                                          agentType=ML_TransformAgent,
                                          model=FFT_Sensor,
                                          sensor_axis=1
                                          )

    minmax_agent = agentNetwork.add_agent(name="MinMaxScaler",
                                          agentType=ML_TransformAgent,
                                          model=MultiMinMaxScaler,
                                          )

    cbae_agent = agentNetwork.add_agent(name="CBAE_Agent", agentType=CBAE_Agent,
                                        conv_architecture=[11, 5, 3],
                                        dense_architecture=[100],
                                        conv_kernel=[100, 50],
                                        conv_stride=[10, 5],
                                        latent_dim=50,
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
                                        return_samples =True,
                                        predict_train = False
                                        )
    sum_propagate_agent = agentNetwork.add_agent(name="SumFeatures", agentType=PropagateTransformAgent, model=sum_nll_features_, use_dmm=False)
    sma_agent = agentNetwork.add_agent(name="SMA", agentType=PropagateTransformAgent, model=sensor_moving_average, window_size=500, use_dmm=False)
    mean_var_bae_agent = agentNetwork.add_agent(name="MeanVarBAESamples", agentType=ML_TransformAgent, model=mean_var_bae_samples, use_dmm=False)
    sum_agent = agentNetwork.add_agent(name="SumFeatures", agentType=ML_TransformAgent,
                                                 model=sum_nll_features, use_dmm=False)
    normalise_agent = agentNetwork.add_agent(name="Normalise", agentType=ML_TransformAgent, model=normalise_1, use_dmm=False)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    plotting_agent = agentNetwork.add_agent("PlottingAgent", agentType=ML_PlottingAgent, model=plot_nll_sensors, use_dmm=False)

    # ood_evaluate_agent = agentNetwork.add_agent(agentType=OOD_EvaluateAgent)

    # connect fft agent to cbae agent
    datastream_agent.bind_output(move_axis_agent, channel=["train","test","dmm_code"])
    datastream_agent.bind_output(plotting_agent, channel="metadata")

    move_axis_agent.bind_output(fft_agent, channel=["train", "test", "dmm_code"])
    fft_agent.bind_output(minmax_agent, channel=["train","test","dmm_code"])
    minmax_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])

    # postprocess cbae outputs v1
    cbae_agent.bind_output(sum_propagate_agent, channel=["test","dmm_code"])
    sum_propagate_agent.bind_output(sma_agent, channel=["test", "dmm_code"])
    sma_agent.bind_output(mean_var_bae_agent, channel=["test","dmm_code"])
    mean_var_bae_agent.bind_output(normalise_agent, channel=["test","dmm_code"])

    # postprocess cbae outputs v2
    # cbae_agent.bind_output(sma_agent, channel=["test","dmm_code"])
    # sma_agent.bind_output(mean_var_bae_agent, channel=["test","dmm_code"])
    # mean_var_bae_agent.bind_output(sum_agent, channel=["test","dmm_code"])
    # sum_agent.bind_output(normalise_agent, channel=["test","dmm_code"])


    # connect to plotting agents
    normalise_agent.bind_output(plotting_agent, channel=["test"])

    # connect evaluate agent
    # normalise_agent.bind_output(plotting_agent, channel=["test"])

    plotting_agent.bind_output(monitor_agent, channel=["plot"])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
