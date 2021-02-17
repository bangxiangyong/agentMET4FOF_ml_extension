import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
from baetorch.baetorch.util.seed import bae_set_seed
from .datastreams import IRIS_Datastream, BOSTON_Datastream
import inspect

# Datastream Agent
from .util.calc_auroc import calc_all_scores


class ML_DatastreamAgent(agentmet4fof_module.AgentMET4FOF):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.

    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"datastream": ["IRIS", "BOSTON"], "train_size": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    parameter_map = {"datastream":{"IRIS": IRIS_Datastream, "BOSTON":BOSTON_Datastream}}
    stylesheet = "triangle"

    def init_parameters(self, datastream="IRIS", train_size=0.8, random_state=123, **data_params):
        self.set_random_state(random_state)
        datastream = self.parameter_map["datastream"][datastream]
        self.datastream = datastream(**data_params)

        self.train_size = train_size

        x_train, x_test, y_train, y_test = train_test_split(self.datastream._quantities, self.datastream._target,
                                                            train_size=self.train_size,
                                                            random_state=random_state)

        self.set_data_sources(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    def set_random_state(self, random_state):
        self.random_state = random_state
        bae_set_seed(random_state)

    def set_data_sources(self, x_train=None, x_test=None, x_ood=None, y_train=None, y_test=None, y_ood=None):
        """
        Set train, test and OOD for x and y's.
        x_ood and y_ood are optional.
        """

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.x_ood = x_ood
        self.y_ood = y_ood

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_output({"quantities":self.x_train, "target":self.y_train},channel="train")
            self.send_output({"quantities":self.x_test, "target":self.y_test},channel="test")
            self.current_state = "Idle"

        elif self.current_state == "Simulate":
            self.send_output({"quantities": self.datastream.next_sample(batch_size=1)}, channel="simulate")

# Transform Agent
class ML_TransformAgent(agentmet4fof_module.AgentMET4FOF):
    """
    A base class for ML_Transform functions, which takes into account the need for fit/transform data
    In `init_parameters`, the `transform_method` is assumed to be a class or function.

    On received message behaviour:
    1a) If it is a class, the agent expects it to have either fit ("train" channel only) or transform method (any channel) on the incoming data.
    1b) Otherwise, the agent will call the function directly on the data (any channel).
    2) Send out the transformed data - {"quantities": iterable, "target": iterable (optional)}

    To inherit:
    1) Set transform_method with initialisation parameters **model_params
    2) Override fit method (model is stored in self.model)
    3) Override transform method (return transformed data with self.model)
    """

    parameter_choices = {"model": ["MinMax", "GP", "MLP"]}
    parameter_map = {"model": {"MinMax": MinMaxScaler, "GP":GaussianProcessClassifier, "MLP":MLPClassifier}}
    stylesheet = "ellipse"

    def init_parameters(self, model=MLPClassifier, supervised = False, random_state=123,  **model_params):
        """
        Initialise model parameters.
        Accepts either a class or a function as model.
        Stores the instantiated model in self.model.

        Parameters
        ----------
        model : class or function

        unsupervised : Boolean. Special keyword for handling fitting to 'target' in the self.fit function

        **model_params : keywords for model parameters instantiation.
        """

        if isinstance(model,str):
            model = self.parameter_map['model'][model]

        self.set_random_state(random_state)
        # assume it as a class model to be initialised
        self.unsupervised = not supervised
        self.model_params = model_params
        self.model = self.instantiate_model(model, model_params)

    def instantiate_model(self, model, model_params):
        # instantiate the model if it is a class
        if hasattr(model, "fit") or inspect.isclass(model):
            new_model = model(**model_params)
        else:
            new_model = model

        return new_model

    def set_random_state(self, random_state):
        self.random_state = random_state
        bae_set_seed(random_state)

    def on_received_message(self, message):
        """
        Handles data from fit/test/simulate channels
        """
        # depending on the channel, we train/test using the message's content.
        channel = message["channel"]
        # extract meta data if available
        if isinstance(message["data"], dict) and "metadata" in message["data"].keys():
            self.metadata = message["data"]["metadata"]

        if channel == "train":
            self.fit(message_data=message["data"])

        if (channel in ["train","test","simulate"]):
            # run prediction/transformation
            transformed_data = self.transform(message["data"])
            output_message = {"quantities": transformed_data}

            # determine if target key is available
            target_available = True if "target" in message["data"].keys() else False
            if target_available:
                output_message.update({"target":message["data"]["target"]})

            # determine if metadata is available
            if hasattr(self, "metadata"):
                output_message.update({"metadata":self.metadata})

            # send output
            self.send_output(output_message, channel=channel)

    def fit(self, message_data):
        """
        Fits self.model on message_data["quantities"]
        """
        if hasattr(self.model, "fit"):
            print("FITTING:"+str(self.name))
            self.model.fit(message_data["quantities"], message_data["target"])

    def transform(self, message_data):
        """
        Transforms and returns message_data["quantities"] using self.model
        If "quantities" is a dict, we apply _transform on every key-val pair
        """

        if isinstance(message_data["quantities"], dict):
            transformed_data = {key:self._transform(val) for key, val in message_data["quantities"].items()}
        else:
            transformed_data = self._transform(message_data["quantities"])

        return transformed_data

    def _transform(self, message_data):
        """
        Internal function. Transforms and returns message_data["quantities"] using self.model
        """

        if hasattr(self.model, "transform"):
            transformed_data = self.model.transform(message_data)
        elif hasattr(self.model, "predict"):
            transformed_data = self.model.predict(message_data)
        else:
            transformed_data = self.model(message_data)

        return transformed_data

class ML_TransformPipelineAgent(ML_TransformAgent):
    """
    Transformer which applies sklearn pipeline style, by combining multiple models in a serial way.

    Need to pass 3 lists: models, superviseds and model params corresponding to the models
    """

    stylesheet = "ellipse"

    def init_parameters(self, models=[MLPClassifier], superviseds = [False], model_params=[], random_state=123):
        """
        Initialise model parameters.
        Accepts either a class or a function as model.
        Stores the instantiated model in self.model.

        Parameters
        ----------
        model : class or function

        unsupervised : Boolean. Special keyword for handling fitting to 'target' in the self.fit function

        **model_params : keywords for model parameters instantiation.
        """

        self.set_random_state(random_state)
        # assume it as a class model to be initialised

        self.model = make_pipeline(*[model(**model_param) for model,model_param in zip(models,model_params)])


class ML_EvaluateAgent(agentmet4fof_module.AgentMET4FOF):
    """
    Last piece in the ML-Pipeline to evaluate the model's performance on the datastream.

    If ml_experiment_proxy is specified, this agent will save the results upon finishing.

    Use this in the conventional supervised sense.
    """
    parameter_choices = {"evaluate_method": ["f1_score", "rmse"], "average":["micro"]}
    parameter_map = {"evaluate_method": {"f1_score": f1_score, "rmse":mean_squared_error}}


    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None, send_plot=True,  **evaluate_params):
        evaluate_method_ = self.parameter_map["evaluate_method"][evaluate_method]
        self.evaluate_method = evaluate_method
        self.ml_experiment_proxy = ml_experiment_proxy

        self.evaluate_methods = [evaluate_method_]
        self.evaluate_params = [evaluate_params]

    def on_received_message(self, message):
        if message["channel"] == "test":
            y_true = message["data"]["target"]
            y_pred = message["data"]["quantities"]

            results = {evaluate_method.__name__:evaluate_method(y_true, y_pred, **evaluate_param)
                       for evaluate_method,evaluate_param in zip(self.evaluate_methods,self.evaluate_params)}
            self.log_info(str(results))
            self.upload_result(results)

            # regression
            if self.evaluate_method == "rmse":
                graph_comparison = self.plot_comparison(y_true, y_pred,
                                                        from_agent=message['from'],
                                                        sum_performance="RMSE: " + str(results['mean_squared_error']))
                self.send_plot(graph_comparison)

    def upload_result(self, results):
        if self.ml_experiment_proxy is not None:
            for key,val in results.items():
                self.ml_experiment_proxy.upload_result(results={key:val})
                self.ml_experiment_proxy.save_result()
            self.log_info("Saved results")

    def plot_comparison(self, y_true, y_pred, from_agent = "", sum_performance= ""):

        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots()
        ax.scatter(y_true,y_pred)
        fig.suptitle("Prediction vs True Label: " + from_agent)
        ax.set_title(sum_performance)
        ax.set_xlabel("Y True")
        ax.set_ylabel("Y Pred")
        return fig


class ML_AggregatorAgent(agentmet4fof_module.AgentMET4FOF):
    """
    Syncs and concatenate the data from its Input Agents
    """
    parameter_choices = {}

    def init_parameters(self):
        self.train_size = None
        self.test_size = None

    def on_received_message(self, message):
        if self.current_state == "Running":
            #update buffer with received data from input agent
            #By default, the AgentBuffer is a FIFO buffer and when new n entries are added to a filled buffer,
            #n entries from the left of buffer will be automatically removed.

            # store data from input agents as list of 'packets'
            # hence each index of the list will be used to sync the data entries
            self.buffer.store(agent_from=message['from'], data=[message['data']])

            buffer_keys = list(self.buffer.keys())
            input_agent_keys = list(self.Inputs.keys())
            sync_counter = 0
            for input_agent in input_agent_keys:
                if (input_agent in buffer_keys) and (len(self.buffer[input_agent]) >0):
                    sync_counter += 1
            if sync_counter == len(input_agent_keys):
                popped_data = self.buffer.popleft(n=1)
                print(popped_data)
                # concatenate the popped data

                # concat_quantities = [popped_data[agent]['quantities'] for agent in popped_data.keys()]
                output_final = {}
                concat_quantities = list(popped_data.values())
                first_entry = concat_quantities[0][0] # {"quantities":[]}
                if isinstance(first_entry["quantities"], dict):
                    full_quantities = {key:np.concatenate([quantity[0]["quantities"][key] for quantity in concat_quantities], axis=-1)
                                       for key in first_entry["quantities"].keys()}
                else:
                    full_quantities = np.concatenate([quantity[0]["quantities"] for quantity in concat_quantities], axis=-1)
                # self.send_output({'quantities':buffer_mean, 'time':buffer_content['time'][-1]})
                output_final.update({"quantities":full_quantities})
                if isinstance(first_entry,dict) and ("target" in first_entry.keys()):
                    full_target = first_entry["target"]
                    output_final.update({"target": full_target})
                if isinstance(first_entry, dict) and ("metadata" in first_entry.keys()):
                    metadata = first_entry["metadata"]
                    output_final.update({"metadata": metadata})
                self.send_output(data=output_final, channel=message["channel"])

class OOD_EvaluateAgent(ML_EvaluateAgent):
    """
    Last piece in the ML-Pipeline to evaluate the model's performance on the datastream.
    If ml_experiment_proxy is specified, this agent will save the results upon finishing.

    """
    parameter_choices = {}

    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None, **evaluate_params):
        self.ml_experiment_proxy = ml_experiment_proxy

    def on_received_message(self, message):

        if message["channel"] == "test":
            message_data_quantities = message["data"]["quantities"]
            nll_test_mu = message_data_quantities["test"]["nll_mu"]
            nll_test_var = message_data_quantities["test"]["nll_var"]
            y_test_var = message_data_quantities["test"]["y_var"]
            enc_test_var = message_data_quantities["test"]["enc_var"]

            nll_ood_mu = message_data_quantities["ood"]["nll_mu"]
            nll_ood_var = message_data_quantities["ood"]["nll_var"]
            y_ood_var = message_data_quantities["ood"]["y_var"]
            enc_ood_var = message_data_quantities["ood"]["enc_var"]

            max_magnitude = 1000000
            auroc_score_nllmu, gmean_nllmu, aps_nllmu, tpr_new_nllmu, fpr_new_nllmu = calc_all_scores(nll_test_mu.mean(-1), nll_ood_mu.mean(-1))
            auroc_score_nllvar, gmean_nllvar, aps_nllvar, tpr_new_nllvar, fpr_new_nllvar = calc_all_scores(
                np.clip(nll_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(nll_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_yvar, gmean_yvar, aps_yvar, tpr_new_yvar, fpr_new_yvar = calc_all_scores(
                np.clip(y_test_var.mean(-1), -max_magnitude, max_magnitude),
                np.clip(y_ood_var.mean(-1), -max_magnitude, max_magnitude))
            auroc_score_enc_var, gmean_enc_var, aps_enc_var, tpr_new_enc_var, fpr_new_enc_var = calc_all_scores(
                np.sum(enc_test_var[1], -1), np.sum(enc_ood_var[1], -1))

            score_nll_mu = {"auroc": auroc_score_nllmu, "gmean": gmean_nllmu, "aps": aps_nllmu,
                            "tpr": tpr_new_nllmu, "fpr": fpr_new_nllmu}
            score_nll_var = {"auroc": auroc_score_nllvar, "gmean": gmean_nllvar, "aps": aps_nllvar,
                             "tpr": tpr_new_nllvar, "fpr": fpr_new_nllvar}
            score_y_var = {"auroc": auroc_score_yvar, "gmean": gmean_yvar, "aps": aps_yvar, "tpr": tpr_new_yvar,
                           "fpr": fpr_new_yvar}
            score_enc_var = {"auroc": auroc_score_enc_var, "gmean": gmean_enc_var, "aps": aps_enc_var,
                             "tpr": tpr_new_enc_var, "fpr": fpr_new_enc_var}

            score_nll_mu = {key+"-nll-mu":val for key,val in score_nll_mu.items()}
            score_nll_var = {key + "-nll-var": val for key, val in score_nll_var.items()}
            score_y_var = {key + "-y-var": val for key, val in score_y_var.items()}
            score_enc_var = {key + "-enc-var": val for key, val in score_enc_var.items()}
            results ={}
            for result in [score_nll_mu,score_nll_var, score_y_var, score_enc_var]:
                results.update(result)

            self.log_info(str(results))
            self.upload_result(results)