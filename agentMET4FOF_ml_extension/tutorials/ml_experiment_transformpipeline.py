"""
Now, we demonstrate the use of a ML_TransformPipelineAgent which wraps the sklearn Pipeline.

"""

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork

from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
from agentMET4FOF_ml_extension.ml_agents import ML_DatastreamAgent, ML_TransformPipelineAgent, ML_EvaluateAgent

def main():
    #initialise with dashboard ml experiments
    random_state = 987
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment, backend="mesa")

    datastream_agent = agentNetwork.add_agent(agentType=ML_DatastreamAgent, datastream="IRIS", random_state=random_state)
    pipeline_agent = agentNetwork.add_agent(agentType=ML_TransformPipelineAgent, random_state=random_state,
                                            models=[MinMaxScaler,SVC],
                                            model_params=[{},{"gamma":'auto'}]
                                            )
    evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="f1_score", average="micro")

    #bind outputs
    datastream_agent.bind_output(pipeline_agent)
    pipeline_agent.bind_output(evaluator_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()

