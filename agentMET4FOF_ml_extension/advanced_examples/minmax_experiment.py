import numpy as np
from sklearn.linear_model import BayesianRidge

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.advanced_examples.supervised_zema_emc_baseline import clip01
from agentMET4FOF_ml_extension.evaluate_agents import EvaluateSupervisedUncAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_TransformPipelineAgent, ML_EvaluateAgent, \
    ML_PlottingAgent, ML_InverseTransformAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagatePipelineAgent, PropagateTransformAgent, \
    PropagateInverseTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.bfc import FFT_BFC
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateAgent
from agentMET4FOF_ml_extension.util.helper import compute_mean_std, move_axis
from agentMET4FOF_ml_extension.util.pearson_fs import Pearson_FeatureSelection
from agentMET4FOF_ml_extension.util.plot_samples import plot_time_series_samples, plot_uncertain_time_series
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
import plotly.graph_objs as go
np.random.seed(100)

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread


def main():
    random_state = 123
    use_dmm = True

    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              train_axis=[5],
                                              test_axis=[5],
                                              train_size=0.5,
                                              random_state=random_state,
                                              use_dmm=use_dmm,
                                              send_test_samples =True,
                                              shuffle=True)
    pre_minmax_agent = agentNetwork.add_agent(name="Pre-MinMax", agentType=ML_TransformAgent,
                                              model=MultiMinMaxScaler, send_train_model=True,
                                              send_test_samples=True,
                                              use_dmm=use_dmm, random_state=random_state,
                                              clip=False)
    post_minmax_agent = agentNetwork.add_agent(name="Post-InverseMinMax", agentType=ML_InverseTransformAgent,
                                               model=MultiMinMaxScaler,
                                               random_state=random_state,
                                               send_test_samples =True,
                                               use_dmm=use_dmm)
    plot_sample_pre_agent = agentNetwork.add_agent(name="Plot-Samples-pre",agentType=ML_PlottingAgent, model=plot_time_series_samples,
                                               sensor_axis=2)
    plot_sample_post_agent = agentNetwork.add_agent(name="Plot-Samples-post",agentType=ML_PlottingAgent, model=plot_time_series_samples,
                                               sensor_axis=2)
    monitor_samples_agent = agentNetwork.add_agent(name="Monitor-Samples",agentType=MonitorAgent)

    agentNetwork.bind_agents(datastream_agent,pre_minmax_agent, channel=["train","test"])
    agentNetwork.bind_agents(pre_minmax_agent,post_minmax_agent, channel=["trained_model","test"])
    agentNetwork.bind_agents(pre_minmax_agent,plot_sample_pre_agent, channel=["test"])

    agentNetwork.bind_agents(datastream_agent, plot_sample_post_agent, channel=["metadata"])
    agentNetwork.bind_agents(datastream_agent, plot_sample_pre_agent, channel=["metadata"])

    agentNetwork.bind_agents(post_minmax_agent,plot_sample_post_agent, channel=["test_samples"])
    agentNetwork.bind_agents(plot_sample_post_agent,monitor_samples_agent, channel=["plot"])
    agentNetwork.bind_agents(plot_sample_pre_agent, monitor_samples_agent, channel=["plot"])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
