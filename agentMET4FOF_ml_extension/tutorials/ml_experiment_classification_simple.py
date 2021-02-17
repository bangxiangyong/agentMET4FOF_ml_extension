"""
This shows a basic setup of an agent pipeline for classification using the Boston dataset from sklearn.
"""

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork

from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
from agentMET4FOF_ml_extension.ml_agents import ML_DatastreamAgent, ML_TransformAgent, ML_EvaluateAgent

def main():
    #initialise with dashboard ml experiments
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment, backend="mesa")

    datastream_agent = agentNetwork.add_agent(agentType=ML_DatastreamAgent, datastream="IRIS")
    gp_agent = agentNetwork.add_agent(agentType=ML_TransformAgent, model="GP")
    evaluator_agent = agentNetwork.add_agent(agentType=ML_EvaluateAgent, evaluate_method="f1_score", average="micro")

    #bind outputs

    datastream_agent.bind_output(gp_agent)
    gp_agent.bind_output(evaluator_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
