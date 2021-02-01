import os
import pickle
from datetime import datetime

from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# from agentMET4FOF.agentMET4FOF.agents import AgentMET4FOF, Coalition
import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
from agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF

import pandas as pd

class IRIS_Datastream(DataStreamMET4FOF):
    def __init__(self):
        iris = datasets.load_iris()
        self.set_data_source(quantities=iris.data, target=iris.target)

# Datastream Agent
class ML_DatastreamAgent(agentmet4fof_module.AgentMET4FOF):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.

    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"datastream": ["IRIS"], "train_size": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}
    parameter_map = {"datastream":{"IRIS":IRIS_Datastream}}

    def init_parameters(self, random_state=123, datastream="IRIS", train_size=0.8, **data_params):
        datastream = self.parameter_map["datastream"][datastream]
        self.datastream = datastream(**data_params)

        self.train_size = train_size
        self.random_state = random_state
        x_train, x_test, y_train, y_test = train_test_split(self.datastream.quantities, self.datastream.target,
                                                            train_size=self.train_size,
                                                            random_state=random_state)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

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

    def init_parameters(self, model=MinMaxScaler, unsupervised=False, **model_params):
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
        model = self.parameter_map['model'][model]

        # assume it as a class model to be initialised
        self.unsupervised = unsupervised

        if hasattr(model, "fit"):
            self.model = model(**model_params)
        else:
            self.model = model

    def on_received_message(self, message):
        """
        Handles data from fit/test/simulate channels
        """
        # depending on the channel, we train/test using the message's content.
        channel = message["channel"]
        if channel == "train":
            self.fit(message_data=message["data"])

        if (channel in ["train","test","simulate"]):
            # run prediction/transformation
            transformed_data = self.transform(message["data"])

            # send output
            target_available = True if "target" in message["data"].keys() else False
            if target_available:
                self.send_output({"quantities":transformed_data, "target":message["data"]["target"]}, channel=channel)
            else:
                self.send_output({"quantities":transformed_data}, channel=channel)

    def fit(self, message_data):
        """
        Fits self.model on message_data["quantities"]
        """
        if hasattr(self.model, "fit"):
            if self.unsupervised:
                self.model.fit(message_data["quantities"])
            else:
                self.model.fit(message_data["quantities"], message_data["target"])

    def transform(self, message_data):
        """
        Transforms and returns message_data["quantities"] using self.model
        """

        if hasattr(self.model, "transform"):
            transformed_data = self.model.transform(message_data["quantities"])
        elif hasattr(self.model, "predict"):
            transformed_data = self.model.predict(message_data["quantities"])
        else:
            transformed_data = self.model(message_data["quantities"])
        return transformed_data

class ML_EvaluateAgent(agentmet4fof_module.AgentMET4FOF):
    """
    Last piece in the ML-Pipeline to evaluate the model's performance on the datastream.

    If ml_experiment_proxy is specified, this agent will save the results upon finishing.
    """
    parameter_choices = {"evaluate_method": ["f1_score"], "average":["micro"]}
    parameter_map = {"evaluate_method": {"f1_score": f1_score}}

    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None, **evaluate_params):
        evaluate_method = self.parameter_map["evaluate_method"][evaluate_method]

        self.ml_experiment_proxy = ml_experiment_proxy

        self.evaluate_methods = [evaluate_method]
        self.evaluate_params = [evaluate_params]

    def on_received_message(self, message):
        if message["channel"] == "test":
            results = {evaluate_method.__name__:evaluate_method(message["data"]["target"], message["data"]["quantities"], **evaluate_param)
                       for evaluate_method,evaluate_param in zip(self.evaluate_methods,self.evaluate_params)}
            self.log_info(str(results))
            self.upload_result(results)

    def upload_result(self, results):
        if self.ml_experiment_proxy is not None:
            for key,val in results.items():
                self.ml_experiment_proxy.upload_result(results={key:val})
                self.ml_experiment_proxy.save_result()









