import numpy as np
from sklearn.ensemble import RandomForestRegressor
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
    use_dmm= True

    # start agent network server
    agentNetwork = AgentNetwork(log_filename=False, backend="mesa")

    # init agents by adding into the agent network
    datastream_agent = agentNetwork.add_agent(agentType=ZEMA_DatastreamAgent,
                                              train_axis=[3],
                                              test_axis=[3,5,7],
                                              train_size=0.8,
                                              random_state=random_state,
                                              shuffle=True)

    pipeline_agent = agentNetwork.add_agent(name="Supervised-Pipeline", agentType=ML_TransformPipelineAgent, random_state=random_state,
                                                      pipeline_models=[FFT_BFC, Pearson_FeatureSelection, BayesianRidge],
                                                      pipeline_params=[{},{},{}], use_dmm=use_dmm
                                                      )

    postproc_agent = agentNetwork.add_agent(name="Clipping-0-100",agentType=ML_TransformAgent, model= clip01, use_dmm=use_dmm)
    evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="rmse")
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    # =========SUPERVISED PIPELINE===========
    datastream_agent.bind_output(pipeline_agent,channel=["train","test","dmm_code"]) # without BAE
    pipeline_agent.bind_output(postproc_agent, channel=["test","dmm_code"])
    postproc_agent.bind_output(evaluator_agent, channel=["test"])
    evaluator_agent.bind_output(monitor_agent, channel="plot")

    # connect evaluate agent to cbae
    postproc_agent.bind_output(evaluator_agent)

    agentNetwork.add_coalition(name="Supervised-Model", agents=[pipeline_agent,postproc_agent,evaluator_agent])

    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == '__main__':
    main()
