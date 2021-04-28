import numpy as np
from agentMET4FOF.agentMET4FOF.agents import AgentNetwork

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateRegressionClassificationAgent
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

np.random.seed(100)

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

# set drift
# moving average
# normalise
# evaluate ood

def main():
    random_state = 123
    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              id_axis=[3],
                                              ood_axis=[3],
                                              train_size=0.2,
                                              random_state=random_state,
                                              move_axis=True,
                                              shuffle=False)

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
                                        dmm_samples=1
                                        )

    ood_evaluate_agent = agentNetwork.add_agent(agentType=OOD_EvaluateRegressionClassificationAgent)

    # connect fft agent to cbae agent
    datastream_agent.bind_output(fft_agent)
    fft_agent.bind_output(minmax_agent)
    minmax_agent.bind_output(cbae_agent)

    # connect evaluate agent to cbae
    cbae_agent.bind_output(ood_evaluate_agent)

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
