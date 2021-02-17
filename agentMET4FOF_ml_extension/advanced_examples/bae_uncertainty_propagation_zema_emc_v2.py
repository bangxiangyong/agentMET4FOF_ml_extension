import numpy as np
from sklearn.linear_model import BayesianRidge

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_TransformPipelineAgent, ML_EvaluateAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagatePipelineAgent, PropagateTransformAgent, \
    PropagateInverseTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.bfc import FFT_BFC
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateAgent
from agentMET4FOF_ml_extension.util.helper import move_axis, clip01
from agentMET4FOF_ml_extension.util.pearson_fs import Pearson_FeatureSelection
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

np.random.seed(100)

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

def main():
    random_state = 123
    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              train_axis=[5],
                                              test_axis=[3,5,7],
                                              train_size=0.8,
                                              random_state=random_state,
                                              move_axis=False,
                                              shuffle=True)

    pre_moveaxis_agent = agentNetwork.add_agent(name="Pre-MoveXis", agentType=ML_TransformAgent, model=move_axis, predict_train=True)
    pre_minmax_agent = agentNetwork.add_agent(name="Pre-MinMax", agentType=ML_TransformAgent,
                                              model=MultiMinMaxScaler, send_train_model=True, clip=False)
    cbae_agent = agentNetwork.add_agent(name="CBAE", agentType=CBAE_Agent,
                                        conv_architecture=[11, 5, 3],
                                        dense_architecture=[100],
                                        conv_kernel=[200, 100],
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
                                        dmm_samples=3,
                                        return_samples = True,
                                        )

    post_minmax_agent = agentNetwork.add_agent(name="Post-InverseMinMax", agentType=PropagateInverseTransformAgent, model=MultiMinMaxScaler, num_samples=1, return_mean=False, first_axis=2, second_axis=3)
    post_moveaxis_agent = agentNetwork.add_agent(name="Post-MoveXis", agentType=PropagateTransformAgent, model=move_axis, num_samples=1, return_mean=False, first_axis=2, second_axis=3)

    propagate_pipeline_agent = agentNetwork.add_agent(name="Supervised-Pipeline", agentType=PropagatePipelineAgent, random_state=random_state,
                                                      models=[FFT_BFC, Pearson_FeatureSelection, BayesianRidge], num_samples=2,
                                                      model_params=[{},{},{}]
                                                      )

    postproc_agent = agentNetwork.add_agent(name="Clipping-0-100",agentType=ML_TransformAgent, model= clip01)
    evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="rmse")
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # connect fft agent to cbae agent

    #==========CBAE COALITION============
    datastream_agent.bind_output(pre_moveaxis_agent, channel=["train","test"])
    pre_moveaxis_agent.bind_output(pre_minmax_agent, channel=["train", "test"])
    pre_minmax_agent.bind_output(cbae_agent, channel=["train", "test"])
    pre_minmax_agent.bind_output(post_minmax_agent, channel=["trained_model"])

    # without min max inverse
    cbae_agent.bind_output(post_moveaxis_agent, channel=["train", "test"])

    # with min max inverse
    # cbae_agent.bind_output(post_minmax_agent, channel=["train","test"])
    # post_minmax_agent.bind_output(post_moveaxis_agent, channel=["train", "test"])

    post_moveaxis_agent.bind_output(propagate_pipeline_agent, channel=["train", "test"])

    # =========SUPERVISED PIPELINE===========
    # datastream_agent.bind_output(propagate_pipeline_agent,channel=["train","test"]) # without BAE
    propagate_pipeline_agent.bind_output(postproc_agent, channel="test")
    postproc_agent.bind_output(evaluator_agent, channel="test")
    evaluator_agent.bind_output(monitor_agent, channel="plot")

    # connect evaluate agent to cbae
    postproc_agent.bind_output(evaluator_agent)

    agentNetwork.add_coalition(name="CBAE-Processing", agents=[pre_moveaxis_agent,pre_minmax_agent,cbae_agent,post_minmax_agent,post_moveaxis_agent])
    agentNetwork.add_coalition(name="Propagate-Supervised", agents=[propagate_pipeline_agent,postproc_agent,evaluator_agent])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
