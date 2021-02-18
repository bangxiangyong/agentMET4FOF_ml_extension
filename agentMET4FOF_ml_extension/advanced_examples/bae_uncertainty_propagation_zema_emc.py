import numpy as np
from sklearn.linear_model import BayesianRidge

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent
from agentMET4FOF_ml_extension.advanced_examples.supervised_zema_emc_baseline import clip01

from agentMET4FOF_ml_extension.ml_agents import ML_TransformAgent, ML_TransformPipelineAgent, ML_EvaluateAgent
from agentMET4FOF_ml_extension.bae_agents import CBAE_Agent, PropagatePipelineAgent, PropagateTransformAgent, \
    PropagateInverseTransformAgent
from agentMET4FOF_ml_extension.datastream_agents import ZEMA_DatastreamAgent
from agentMET4FOF_ml_extension.util.bfc import FFT_BFC
from agentMET4FOF_ml_extension.util.fft_sensor import FFT_Sensor
from agentMET4FOF_ml_extension.ood_evaluate_agents import OOD_EvaluateAgent
from agentMET4FOF_ml_extension.util.pearson_fs import Pearson_FeatureSelection
from baetorch.baetorch.util.minmax import MultiMinMaxScaler

np.random.seed(100)

import matplotlib
matplotlib.use('Agg') # https://stackoverflow.com/questions/27147300/matplotlib-tcl-asyncdelete-async-handler-deleted-by-the-wrong-thread

def clip01(x_test, min=0, max=100):
    return np.clip(x_test,min, max)

def move_axis(x, first_axis=1,second_axis=2):
    return np.moveaxis(x,first_axis,second_axis)

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
                                              shuffle=True)

    pre_moveaxis_agent = agentNetwork.add_agent(name="Pre-MoveXis", agentType=ML_TransformAgent,
                                                model=move_axis, predict_train=True, use_dmm=use_dmm, random_state=random_state)
    pre_minmax_agent = agentNetwork.add_agent(name="Pre-MinMax", agentType=ML_TransformAgent,
                                              model=MultiMinMaxScaler, send_train_model=True, use_dmm=use_dmm, random_state=random_state,
                                              clip=False)

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
                                        num_epochs=2,
                                        use_dmm=True,
                                        dmm_samples=2,
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

    propagate_pipeline_agent = agentNetwork.add_agent(name="Supervised-Pipeline", agentType=PropagatePipelineAgent,
                                                      random_state=random_state,
                                                      pipeline_models=[FFT_BFC, Pearson_FeatureSelection, BayesianRidge],
                                                      single_model=False,
                                                      pipeline_params=[{},{},{}],
                                                      use_dmm=use_dmm,
                                                      )

    postproc_agent = agentNetwork.add_agent(name="Clipping-0-100",agentType=PropagateTransformAgent,
                                            random_state=random_state,
                                            model= clip01,
                                            use_dmm=use_dmm,
                                            single_model=True,
                                            return_mean=True)

    evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="rmse")
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # connect fft agent to cbae agent

    #==========CBAE PREPROCESSING============
    datastream_agent.bind_output(pre_moveaxis_agent, channel=["train","test","dmm_code"])
    pre_moveaxis_agent.bind_output(pre_minmax_agent, channel=["train", "test","dmm_code"])
    pre_minmax_agent.bind_output(cbae_agent, channel=["train", "test","dmm_code"])
    pre_minmax_agent.bind_output(post_minmax_agent, channel=["trained_model","dmm_code"])

    # without min max inverse
    cbae_agent.bind_output(post_moveaxis_agent, channel=["train", "test","dmm_code"])

    # with min max inverse
    # cbae_agent.bind_output(post_minmax_agent, channel=["train","test","dmm_code"])
    # post_minmax_agent.bind_output(post_moveaxis_agent, channel=["train", "test","dmm_code"])

    post_moveaxis_agent.bind_output(propagate_pipeline_agent, channel=["train", "test","dmm_code"])

    # =========SUPERVISED PIPELINE===========
    propagate_pipeline_agent.bind_output(postproc_agent, channel=["test","dmm_code"])

    postproc_agent.bind_output(evaluator_agent, channel="test")
    evaluator_agent.bind_output(monitor_agent, channel="plot")

    # ===========SETUP COALITIONS============
    agentNetwork.add_coalition(name="CBAE-Processing", agents=[pre_moveaxis_agent,pre_minmax_agent,cbae_agent,post_minmax_agent,post_moveaxis_agent])
    agentNetwork.add_coalition(name="Propagate-Supervised", agents=[propagate_pipeline_agent,postproc_agent,evaluator_agent])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
