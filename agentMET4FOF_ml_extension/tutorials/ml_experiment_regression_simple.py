"""
This shows a basic setup of an agent pipeline for regression using the Boston dataset from sklearn.
"""


from sklearn.ensemble import RandomForestRegressor

from agentMET4FOF.agentMET4FOF.agents import AgentNetwork, MonitorAgent

from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment
from agentMET4FOF_ml_extension.ml_agents import ML_DatastreamAgent, ML_TransformAgent, EvaluateRegressionClassificationAgent

def main():
    #initialise with dashboard ml experiments
    agentNetwork = AgentNetwork(dashboard_extensions=Dashboard_ML_Experiment, backend="mesa")

    datastream_agent = agentNetwork.add_agent(agentType=ML_DatastreamAgent, datastream="BOSTON")
    rf_agent = agentNetwork.add_agent(agentType=ML_TransformAgent, model=RandomForestRegressor)
    evaluator_agent = agentNetwork.add_agent(agentType=EvaluateRegressionClassificationAgent, evaluate_method="rmse")
    monitor_agent = agentNetwork.add_agent(agentType=MonitorAgent)

    #bind outputs
    datastream_agent.bind_output(rf_agent)
    rf_agent.bind_output(evaluator_agent)
    evaluator_agent.bind_output(monitor_agent)

    #set to active running
    agentNetwork.set_running_state()

    # allow for shutting down the network after execution
    return agentNetwork


if __name__ == "__main__":
    main()
