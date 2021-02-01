import agentMET4FOF_ml_extension.agents as ml_agents
from agentMET4FOF.agentMET4FOF.agents import AgentNetwork
from agentMET4FOF_ml_extension.Dashboard_ml_exp import Dashboard_ML_Experiment

def main():
    agentNetwork = AgentNetwork(dashboard_modules=[ml_agents],
                                dashboard_extensions=Dashboard_ML_Experiment,
                                backend="mesa")
if __name__ == "__main__":
    main()
