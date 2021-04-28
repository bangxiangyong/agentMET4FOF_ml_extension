import numpy as np
from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.evaluate_agents import EvaluateAUROCAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent, STRATH_DatastreamAgent
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateRegressionClassificationAgent
from agentMET4FOF_ml_extension.util.helper import move_axis, compute_mean_var
from agentMET4FOF_ml_extension.util.resample_sensor import apply_along_sensor, resample_data
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

np.random.seed(100)

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

# set drift
# moving average
# normalise
# evaluate ood

def main():
    random_state = 999
    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=STRATH_DatastreamAgent,
                                              phase="forging",
                                              train_size=0.5,
                                              random_state=random_state,
                                              shuffle=True)

    resample_agent = agentNetwork.add_agent(name="Resample",
                                            agentType=ML_TransformAgent,
                                            model=apply_along_sensor,
                                            func = resample_data,
                                            sensor_axis = 1,
                                            n=10
                                          )

    minmax_agent = agentNetwork.add_agent(name="MinMaxScaler",
                                          agentType=ML_TransformAgent,
                                          model=MultiMinMaxScaler,
                                          )

    fft_agent = agentNetwork.add_agent(name="FFT_Agent",
                                          agentType=ML_TransformAgent,
                                          model=FFT_Sensor,
                                          sensor_axis=1
                                          )

    cbae_agent = agentNetwork.add_agent(name="CBAE_Agent", agentType=CBAE_Agent,
                                        conv_architecture=[datastream_agent.x_train.shape[-1], 20, 25],
                                        dense_architecture=[100],
                                        conv_kernel=[100, 10], #forging
                                        conv_stride=[2, 2],
                                        # conv_kernel=[500, 50], #forging
                                        # conv_stride=[50, 10],
                                        # conv_kernel=[482, 51], #heating
                                        # conv_stride=[50, 12],
                                        latent_dim=50,
                                        likelihood="homo_gaussian",
                                        learning_rate=0.01,
                                        bae_samples=5,
                                        random_state=random_state,
                                        use_cuda=True,
                                        train_model=True,
                                        num_epochs=150,
                                        move_axis=False,
                                        use_dmm=True,
                                        dmm_samples=1,
                                        return_samples = True
                                        )

    move_axis_agent = agentNetwork.add_agent(name="MoveXis", agentType=ML_TransformAgent, model=move_axis)
    mean_var_agent = agentNetwork.add_agent(name="MeanVar", agentType=ML_TransformAgent, model=compute_mean_var, return_dict=True)

    ood_evaluate_agent = agentNetwork.add_agent(agentType=EvaluateAUROCAgent)
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # connect fft agent to cbae agent
    datastream_agent.bind_output(move_axis_agent, channel=["train","test","dmm_code"])
    move_axis_agent.bind_output(fft_agent, channel=["train","test","dmm_code"])
    fft_agent.bind_output(resample_agent, channel=["train","test","dmm_code"])

    # move_axis_agent.bind_output(resample_agent, channel=["train","test","dmm_code"])
    resample_agent.bind_output(minmax_agent, channel=["train","test","dmm_code"])

    # move_axis_agent.bind_output(minmax_agent, channel=["train","test","dmm_code"])
    minmax_agent.bind_output(cbae_agent, channel=["train","test","dmm_code"])

    # connect evaluate agent to cbae
    cbae_agent.bind_output(mean_var_agent, channel=["test"])
    mean_var_agent.bind_output(ood_evaluate_agent, channel=["test"])
    ood_evaluate_agent.bind_output(monitor_agent, channel=["test","plot"])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
