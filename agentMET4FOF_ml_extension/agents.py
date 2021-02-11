import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
from baetorch.baetorch.util.seed import bae_set_seed
from .datastreams import IRIS_Datastream


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
    parameter_map = {"datastream":{"IRIS": IRIS_Datastream}}
    stylesheet = "triangle"

    def init_parameters(self, datastream="IRIS", train_size=0.8, random_state=123, **data_params):
        self.set_random_state(random_state)
        datastream = self.parameter_map["datastream"][datastream]
        self.datastream = datastream(**data_params)

        self.train_size = train_size

        x_train, x_test, y_train, y_test = train_test_split(self.datastream.quantities, self.datastream.target,
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

    def init_parameters(self, model=MinMaxScaler, unsupervised=False, random_state=123,  **model_params):
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
        self.set_random_state(random_state)
        # assume it as a class model to be initialised
        self.unsupervised = unsupervised

        if hasattr(model, "fit"):
            self.model = model(**model_params)
        else:
            self.model = model

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

    def init_parameters(self, evaluate_method=[], ml_experiment_proxy=None,  **evaluate_params):
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
            self.log_info("Saved results")


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









