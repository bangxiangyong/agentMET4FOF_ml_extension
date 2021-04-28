import os
import pickle
from datetime import datetime
import pandas as pd
import uuid
import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
import agentMET4FOF_ml_extension.ml_agents as ml_agents


class ML_Results :
    """
    Result from the run of an ML Experiment of the following fields:

    run_details : name of experiment and date run
    data_pipeline : details of the data pipeline.
    results : compare results between models

    This will be saved as a pickle to be analysed later on.
    """
    def __init__(self, run_details,data_pipeline_params, results):
        self.run_details = run_details
        self.data_pipeline_params = data_pipeline_params
        self.results = results

    def get_results_pd(self):
        return pd.DataFrame(self.results)

class ML_ExperimentLite:
    """
    Stand alone class for logging ML experiment parameters and their performances.

    Key function is the `save_result` parameterised by `ml_performance` and `ml_parameters`.
    It checks if `ml_parameters` existed, if it is, then we ignore the appending.
    The result is pickled as a pd.DataFrame in the `base_folder`.

    If `ml_exp_name` is set to `auto` on instantiation, we will create a random unique id for it.
    """
    def __init__(self, ml_exp_name = "auto",
                 ml_parameters = {"train_size":0.7, "dataset":"ZEMA", "perturbed_sensors":[0,1,2,3]},
                 base_folder = "MLEXP/",
                 save_csv=True
                 ):
        if ml_exp_name == "auto":
            ml_exp_name = uuid.uuid4().hex
        self.save_csv = save_csv
        self.ml_exp_name = ml_exp_name
        self.ml_parameters = ml_parameters
        self.base_folder = base_folder
        self.file_path = self.base_folder + self.ml_exp_name + ".p"
        self.csv_file_path = self.base_folder + self.ml_exp_name + ".csv"

    def create_results(self, ml_performance:dict, ml_parameters=None):
        ml_results=[]
        for key, val in ml_performance.items():
            temp_ml_results = {"Date":datetime.now()}
            temp_ml_results.update(ml_parameters)
            temp_ml_results.update({"perf_name":key, "perf_score":val})
            ml_results.append(temp_ml_results)
        ml_results = pd.DataFrame(ml_results)
        return ml_results

    def load_results(self):
        # create folder
        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

        # load pickle
        with open(self.file_path, "rb") as f:
            ml_results = pickle.load(f)
        return ml_results

    def save_results(self, ml_performance:dict, ml_parameters=None):
        if ml_parameters is None:
            ml_parameters = self.ml_parameters

        new_ml_results = self.create_results(ml_performance, ml_parameters)

        # create folder
        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

        # check if file exists
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                current_ml_results = pickle.load(f)
                if self.check_entry_exist(current_ml_results, new_ml_results, cols=[]):
                    print("Entry exists...")
                    return 0
                else:
                    new_ml_results = current_ml_results.append(new_ml_results)

        # save pickle
        with open(self.file_path, "wb") as f:
            pickle.dump(new_ml_results, f)
        print("Saving results...")

        # save csv
        if self.save_csv:
            new_ml_results.to_csv(self.csv_file_path, index=False)

    def check_entry_exist(self, main_df, sub_df, cols=[],drop_date=True):
        if len(cols) == 0 :
            cols = main_df.columns
        if drop_date:
            cols = [col for col in cols if col != "Date"]

        temp_main_df = main_df[cols].astype(str)
        temp_sub_df = sub_df[cols].astype(str)

        exists = len(temp_main_df.merge(temp_sub_df)) == len(temp_main_df)
        return exists


class ML_ExperimentCoalition(agentmet4fof_module.Coalition):
    """
    An ML Experiment has a run_date and ml_name identifier.

    It consist of a list of dict results, to be logged/appended by ML evaluator agent
    Every entry consist of:
     1) the pipeline details (data params, model params, randomstate, train_size)
     2) performance method and score (e.g f1-score, 0.97)

    It is foreseen that, we will concatenate the list from multiple experiments into a large dataframe of results,
    which we can then compare model performances.
    """
    def __init__(self, ml_name="simple1", agents=[], random_seed=123, base_folder="MLEXP/"):

        super().__init__(name = ml_name, agents=agents)
        self.base_folder = base_folder
        self.ml_name = ml_name
        self.run_date = datetime.now()
        self.run_details = {"run_name": self.ml_name, "date":self.run_date}
        self.results = []
        self.pipeline_ready = False
        self.random_seed = random_seed
        self.infer_parameters()

    def add_agent(self, agent):
        super().add_agent(agent)
        self.infer_parameters()
        print("PIPELINE READY:" +str(self.pipeline_ready))

    def infer_parameters(self):
        datastream_agent, model_agents, evaluate_agent = self.infer_agents(self.agents)
        if (datastream_agent is not None) and (model_agents is not None) and (evaluate_agent is not None):
            self.data_pipeline_params = self.infer_connections(datastream_agent=datastream_agent,
                                                               model_agents=model_agents,
                                                               evaluate_agent=evaluate_agent,
                                                               random_seed=self.random_seed)
            self.pipeline_ready = True
        else:
            self.pipeline_ready = False

    def infer_agents(self, agents):
        datastream_agent = None
        model_agents = None
        evaluate_agent = None

        for agent in agents:
            if isinstance(agent, ml_agents.ML_DatastreamAgent) or isinstance(agent, ml_agents.ML_AggregatorAgent):
                datastream_agent = agent
            if isinstance(agent, ml_agents.ML_TransformAgent):
                if model_agents is None:
                    model_agents = [agent]
                else:
                    model_agents.append(agent)
            if isinstance(agent, ml_agents.EvaluateRegressionClassificationAgent) or issubclass(type(agent), ml_agents.EvaluateRegressionClassificationAgent):
                evaluate_agent = agent
        return datastream_agent, model_agents, evaluate_agent

    def convert_method_to_name(self, param):
        """
        Extracts name of class
        """
        if isinstance(param, str) or isinstance(param, list) or isinstance(param, float) or isinstance(param, int):
            return param

        else:
            return type(param).__name__

    def infer_connections(self, datastream_agent, model_agents, evaluate_agent, random_seed=123):
        # handle datastream
        datastream_name = datastream_agent.name
        data_params = {param_key:self.convert_method_to_name(datastream_agent.get_attr(param_key)) for param_key, _ in datastream_agent.parameter_choices.items() if hasattr(datastream_agent,"parameter_choices")}

        # append model name
        model_name = model_agents[0].name
        if len(model_agents) >1:
            for model in model_agents:
                model_name = model_name+"->"+model.name
        model_params = [{param_key: self.convert_method_to_name(model_agent.get_attr(param_key)) for param_key, _ in
                       model_agent.parameter_choices.items() if hasattr(model_agent,"parameter_choices")} for model_agent in model_agents]

        # handle evaluator agent
        evaluate_agent.ml_experiment_proxy = self

        # full data pipeline params
        data_pipeline_params = {"data": datastream_name, "data_params": data_params, "model": model_name,
                                     "model_params": model_params, "random_seed": random_seed}
        return data_pipeline_params

    def load_result(self):
        # create folder
        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

        # save pickle
        ml_results = pickle.load(open(self.base_folder+self.ml_name+".p","rb"))
        return ml_results

    def save_result(self):
        """
        This pickles the details of an ML Experiment run.
        An ML Experiment Run consists of the following details:
        run_details : name of experiment and date run
        data_pipeline : details of the data pipeline.
        results : compare results between models
        """

        run_details = pd.DataFrame([self.run_details])
        data_pipeline_params = self.data_pipeline_params
        results = self.results

        ml_results = ML_Results(run_details,data_pipeline_params, results)

        # create folder
        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

        # save pickle
        pickle.dump(ml_results, open(self.base_folder+self.ml_name+".p","wb"))
        print("SAVED RESULT")

    def upload_result(self, results={"perf_name":"perf_score"}, model_name=None):
        """
        A function to append into the ML_Experiment's results list of dicts
        """
        for key,val in results.items():
            new_result_entry = self.data_pipeline_params.copy()
            new_result_entry.update({"perf_name":key, "perf_score":val})

            if model_name is not None:
                new_result_entry.update({"model":model_name})

            self.results.append(new_result_entry)

def add_ml_experiment(agentNetwork, name="MLEXP_1", agents=[], random_seed=123):
    """
    Instantiates a coalition of agents.
    """
    new_ml_exp = ML_ExperimentCoalition(ml_name=name, agents=agents, random_seed=random_seed)
    agentNetwork._get_controller().add_coalition(new_ml_exp)
    return new_ml_exp
