import numpy as np
from sklearn.linear_model import BayesianRidge

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.supervised_zema_emc_baseline import clip01
from agentMET4FOF_ml_extension.evaluate_agents import EvaluateSupervisedUncAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_PlottingAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagatePipelineAgent, PropagateTransformAgent, \
    PropagateInverseTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.bfc import FFT_BFC
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.util.helper import compute_mean_std, move_axis
from agentMET4FOF_ml_extension.util.pearson_fs import Pearson_FeatureSelection
from agentMET4FOF_ml_extension.util.plot_samples import plot_time_series_samples, plot_uncertain_time_series
from baetorch.baetorch.util.minmax import MultiMinMaxScaler
import plotly.graph_objs as go
np.random.seed(100)

# import matplotlib
# matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread


def bar_graph(data, sender_agent):
    """
    Expect data to be double nested dicts:
    First level is the test datasets keys
    But on the second level, it is a list of performance dicts.
    e.g
    {
    "test1":{["rmse":[]]},
    "test2":{["rmse":[]]}
    }

    """
    bar_graphs = []
    for key in data.keys():
        new_bar = go.Bar(name="Axis:"+key, x=list(data[key][-1].keys()), y=list(data[key][-1].values()))
        bar_graphs.append(new_bar)
    return bar_graphs

def main():
    random_state = 123
    use_dmm = True

    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              train_axis=[5],
                                              test_axis=[3,5,7],
                                              train_size=0.5,
                                              random_state=random_state,
                                              use_dmm=use_dmm,
                                              send_test_samples =True,
                                              shuffle=True)
    fft_agent = agentNetwork.add_agent(name="FFT", agentType=ML_TransformAgent, model=FFT_Sensor,
                                       use_dmm=use_dmm, random_state=random_state,
                                       sampling_f=2000, sensor_axis=2, send_test_samples =True,
                                       )
    pre_moveaxis_agent = agentNetwork.add_agent(name="Pre-MoveXis", agentType=ML_TransformAgent,
                                                model=move_axis, predict_train=True, use_dmm=use_dmm, random_state=random_state)
    pre_minmax_agent = agentNetwork.add_agent(name="Pre-MinMax", agentType=ML_TransformAgent,
                                              model=MultiMinMaxScaler, send_train_model=True, use_dmm=use_dmm, random_state=random_state,
                                              clip=False)



    cbae_agent = agentNetwork.add_agent(name="CBAE", agentType=CBAE_Agent,
                                        conv_architecture=[11, 5, 3],
                                        dense_architecture=[100],
                                        conv_kernel=[100, 50],
                                        conv_stride=[10, 5],
                                        latent_dim=50,
                                        likelihood="1_gaussian",
                                        learning_rate=0.01,
                                        bae_samples=1,
                                        random_state=random_state,
                                        use_cuda=True,
                                        train_model=True,
                                        num_epochs=5,
                                        use_dmm=True,
                                        dmm_samples=5,
                                        return_samples = True,
                                        )

    post_minmax_agent = agentNetwork.add_agent(name="Post-InverseMinMax", agentType=PropagateInverseTransformAgent,
                                               model=MultiMinMaxScaler,
                                               random_state=random_state,
                                               single_model=True,
                                               return_mean=False,
                                               use_dmm=use_dmm)

    post_moveaxis_agent = agentNetwork.add_agent(name="Post-MoveXis", agentType=PropagateTransformAgent,
                                                 model=move_axis, single_model=True,
                                                 use_dmm=use_dmm,
                                                 random_state = random_state,
                                                 return_mean=False, first_axis=1,
                                                 second_axis=2)

    bae_compute_mean_std_agent = agentNetwork.add_agent(name="BAE-Compute-MeanStd", agentType=ML_TransformAgent,
                                                 model=compute_mean_std,
                                                 use_dmm=use_dmm,
                                                 random_state = random_state,
                                                 send_test_samples = True, # this enables `test_samples` channel
                                                        example_axis=1,
                                                        )

    supervised_compute_mean_std_agent = agentNetwork.add_agent(name="SV-Compute-MeanStd", agentType=ML_TransformAgent,
                                                 model=compute_mean_std,
                                                 use_dmm=use_dmm,
                                                 random_state = random_state)


    propagate_pipeline_agent = agentNetwork.add_agent(name="Supervised-Pipeline", agentType=PropagatePipelineAgent,
                                                      random_state=random_state,
                                                      pipeline_models=[FFT_BFC, Pearson_FeatureSelection, BayesianRidge],
                                                      single_model=False,
                                                      pipeline_params=[{},{},{}],
                                                      use_dmm=use_dmm,
                                                      )

    clip0100_agent = agentNetwork.add_agent(name="Clipping-0-100",agentType=PropagateTransformAgent,
                                            random_state=random_state,
                                            model= clip01,
                                            use_dmm=use_dmm,
                                            single_model=True,
                                            return_mean=False)

    # evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="rmse")
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluateSupervisedUncAgent)
    monitor_perf_agent = agentNetwork.add_agent(name="Monitor-Performance",agentType=MonitorAgent,
                                                custom_plot_function=bar_graph)
    monitor_samples_agent = agentNetwork.add_agent(name="Monitor-Samples",agentType=MonitorAgent)

    #==========CBAE PREPROCESSING============
    # datastream_agent.bind_output(pre_moveaxis_agent, channel=["train","test","dmm_code"])
    datastream_agent.bind_output(fft_agent, channel=["train", "test", "dmm_code"])
    fft_agent.bind_output(pre_moveaxis_agent, channel=["train","test","dmm_code"])
    pre_moveaxis_agent.bind_output(pre_minmax_agent, channel=["train", "test","dmm_code"])
    pre_minmax_agent.bind_output(post_minmax_agent, channel=["trained_model","dmm_code"])
    pre_minmax_agent.bind_output(cbae_agent, channel=["train", "test","dmm_code"])

    # without min max inverse
    # cbae_agent.bind_output(post_moveaxis_agent, channel=["train", "test","dmm_code"])

    # with min max inverse
    cbae_agent.bind_output(post_minmax_agent, channel=["train","test","dmm_code"])
    post_minmax_agent.bind_output(post_moveaxis_agent, channel=["train", "test","dmm_code"])

    # post_moveaxis_agent.bind_output(propagate_pipeline_agent, channel=["train", "test","dmm_code"])

    # =========SUPERVISED PIPELINE===========
    propagate_pipeline_agent.bind_output(clip0100_agent, channel=["test","dmm_code"])
    clip0100_agent.bind_output(supervised_compute_mean_std_agent, channel="test")
    supervised_compute_mean_std_agent.bind_output(evaluator_agent, channel="test")

    evaluator_agent.bind_output(monitor_perf_agent, channel=["default","plot"])

    #===========PLOTTING AGENTS=================
    plot_sample_agent = agentNetwork.add_agent(name="Plot-Samples",agentType=ML_PlottingAgent, model=plot_time_series_samples,
                                               sensor_axis=2)
    plot_recon_agent = agentNetwork.add_agent(name="Plot-BAE-Recon",agentType=ML_PlottingAgent, model=plot_uncertain_time_series,
                                               sensor_axis=2)

    # recon samples
    post_moveaxis_agent.bind_output(bae_compute_mean_std_agent, channel=["test"])
    bae_compute_mean_std_agent.bind_output(plot_recon_agent, channel=["test_samples"])

    # raw samples
    agentNetwork.bind_agents(datastream_agent,plot_sample_agent,channel=["metadata"])
    agentNetwork.bind_agents(fft_agent, plot_sample_agent, channel=["test_samples"])

    agentNetwork.bind_agents(datastream_agent,plot_recon_agent,channel=["metadata"])

    # connect plotting agents to monitors
    plot_recon_agent.bind_output(monitor_samples_agent, channel="plot")
    plot_sample_agent.bind_output(monitor_samples_agent, channel="plot")

    # ===========SETUP COALITIONS============
    agentNetwork.add_coalition(name="CBAE-Processing", agents=[pre_moveaxis_agent,pre_minmax_agent,cbae_agent,post_minmax_agent,post_moveaxis_agent, bae_compute_mean_std_agent])
    agentNetwork.add_coalition(name="Propagate-Supervised", agents=[propagate_pipeline_agent,clip0100_agent,supervised_compute_mean_std_agent, evaluator_agent])
    agentNetwork.add_coalition(name="Plots", agents=[plot_sample_agent,plot_recon_agent, monitor_samples_agent,monitor_perf_agent])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
