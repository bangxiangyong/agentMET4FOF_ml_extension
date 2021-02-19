import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.neural_network import MLPClassifier

import agentMET4FOF.agentMET4FOF.agents as agentmet4fof_module
from baetorch.baetorch.util.data_model_manager import DataModelManager
from baetorch.baetorch.util.seed import bae_set_seed
from .datastreams import IRIS_Datastream, BOSTON_Datastream
import inspect
import functools

# Datastream Agent
from .util.calc_auroc import calc_all_scores

class ML_BaseAgent(agentmet4fof_module.AgentMET4FOF):
    """
    Abstract base class with handy methods for all inherited agents.

    Such methods include DataModelManager methods, setting random state,
    """

    def init_dmm(self, use_dmm):
        """
        Call this in the init_parameter, after storing local parameters
        which form the signature of the agent.

        """
        self.use_dmm = use_dmm
        if use_dmm:
           self.dmm = DataModelManager()
           self.dmm_code = self.encode_current_params()

    def get_current_params(self):
        variables = str({key: val.__name__ if (inspect.isfunction(val) or inspect.ismethod(val) or inspect.isclass(val))
        else str(val) for key, val in self.__dict__.items()
                         if
                         key not in ["self", "dmm", "use_dmm", "mesa_message_queue", "model", "mesa_model", "backend", "forward_model","bae_model"]})
        print("-------------------------")
        print(self.name)
        print(variables)
        return variables

    def encode_current_params(self):
        return self.dmm.encode(self.get_current_params(),return_pickle=False, return_compact=True)

    def set_random_state(self, random_state):
        self.random_state = random_state
        bae_set_seed(random_state)

    def send_dmm_code(self):
        # sends out dmm code if available
        self.send_output({"dmm_code":self.dmm_code}, channel="dmm_code")

    def mix_dmm_code(self, current_code, received_code):
        new_dmm_code = self.dmm.encode(datatype=current_code + received_code, return_pickle=False, return_compact=True)
        return new_dmm_code

    def dmm_wrap(self, wrap_method, additional_id="", *args, **kwargs):
        """
        Convenient method for wrapping the DMM around a method with self checking on whether the self.use_dmm parameter is used.
        """

        if self.use_dmm:
            return self.dmm.wrap(wrap_method, additional_id+self.dmm_code, *args, **kwargs)
        else:
            return wrap_method(*args, **kwargs)

class ML_DatastreamAgent(ML_BaseAgent):
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

    def init_parameters(self, datastream="IRIS", train_size=0.8, random_state=123, use_dmm=False, **datastream_params):
        """
        Initialises important parameters of a base class of ML Datastream
        Users looking to override this should call this on super() to ensure parameters such as `use_dmm`, `random_state`
        and `train_size` are filled in.

        If a custom procedure is required for setting up the datastream (other than the basic one provided here),
        set the datastream parameter to be `None` and proceed in setting up the custom datastream.

        """
        # if use Data Model Manager is enabled
        self.set_random_state(random_state)
        self.train_size = train_size
        self.init_dmm(use_dmm)

        if datastream is not None:
            if isinstance(datastream, str):
                # resort to default provided datasets
                datastream = self.parameter_map["datastream"][datastream]
            self.datastream = datastream(**datastream_params)

            x_train, x_test, y_train, y_test = train_test_split(self.datastream._quantities, self.datastream._target,
                                                                train_size=self.train_size,
                                                                random_state=random_state)
            self.set_data_sources(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)


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
            # if use dmm, we send out the dmm code first
            if self.use_dmm:
                self.send_dmm_code()

            self.send_output({"quantities":self.x_train, "target":self.y_train},channel="train")
            self.send_output({"quantities":self.x_test, "target":self.y_test},channel="test")
            self.current_state = "Idle"

        elif self.current_state == "Simulate":
            self.send_output({"quantities": self.datastream.next_sample(batch_size=1)}, channel="simulate")


# Transform Agent
class ML_TransformAgent(ML_BaseAgent):
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

    def init_parameters(self, model=MLPClassifier, random_state=123,
                        predict_train=True,
                        send_train_model=False, use_dmm=False, **model_params):
        """
        Initialise model parameters.
        Accepts either a class or a function as model.
        Stores the instantiated model in self.model.

        Parameters
        ----------
        model : class or function

        predict_train : boolean
            After fitting model, this determines whether to apply `predict` on the training data and then `send_output`.
            If we need to propagate the transformed trained data (e.g normalised data), then we should set to `True`, otherwise it is unnecessary and more
            computationally efficient to set to `False`.

        send_train_model : boolean
            After fitting model, this parameter determines whether to send out the trained model on the channel "trained_model"
            On the receiving agent, it can now equip this model to conduct transforms or inverse transforms.

        use_dmm : boolean
            Specify whether to use DataModelManager (DMM) in keeping track of the pipeline which has been executed before.
            If fitting/transforming has been executed, this loads the transformed data or fitted model from the DMM.
            Firstly it keeps track of the DMM Code which is an encoded form of the agent's parameters
            and propagates in the `send_output` with channel of "dmm_code".
            Receiving agents will `mix` the new received DMM Code with its own DMM Code and propagate the code forward.
            Due to this, agents have to connect the "dmm_code" channel to fully make use of the `use_dmm` function

        **model_params : keywords for model parameters instantiation.
        """
        # if use Data Model Manager is enabled

        if isinstance(model,str):
            self.model_class = self.parameter_map['model'][model]
        elif model is not None:
            self.model_class = model

        self.set_random_state(random_state)
        self.predict_train = predict_train

        # assume it as a class model to be initialised
        self.model_params = model_params
        self.send_train_model = send_train_model
        self.init_dmm(use_dmm)

        if model is not None:
            self.forward_model = self.instantiate_model()

    def instantiate_model(self):
        model = self.model_class
        model_params = self.model_params

        # instantiate the model if it is a class
        if inspect.isclass(model):
            new_model = model(**model_params)
        # assume it as a function
        elif inspect.isfunction(model) or inspect.ismethod(model):
            new_model = functools.partial(model, **model_params)
        # it is an already instantiated model
        else:
            new_model = model
        return new_model

    def on_received_message(self, message):
        """
        Handles data from fit/test/simulate channels
        """
        # depending on the channel, we train/test using the message's content.
        channel = message["channel"]

        self.handle_channels(message, channel)

        if (channel in ["train","test","simulate"]):
            # do not proceed to applying predict on train data
            # if channel is "train" but predict_train
            if channel == "train" and not self.predict_train:
                return 0

            # run prediction/transformation
            # wrap predictions method with dmm if enabled
            transformed_data = self.dmm_wrap(self.transform, message_data=message["data"], additional_id=channel)

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

    def handle_channels(self, message, channel):
        # extract meta data if available
        if isinstance(message["data"], dict) and "metadata" in message["data"].keys():
            self.metadata = message["data"]["metadata"]

        if channel == "metadata":
            self.metadata = message["data"]

        # update dmm code and propagate forward
        elif channel == "dmm_code" and self.use_dmm:
            self.dmm_code = self.mix_dmm_code(self.dmm_code, message["data"]["dmm_code"])
            self.send_dmm_code()

        elif channel == "trained_model":
            self.forward_model = message["data"]["model"]

        elif channel == "train":
            # wrap fit method with dmm if enabled
            self.forward_model = self.dmm_wrap(self.fit, message_data=message["data"], additional_id=channel)

            if hasattr(self, "send_train_model") and self.send_train_model:
                self.send_output({"model":self.forward_model}, channel="trained_model")


    def fit(self, message_data):
        """
        Fits self.model on message_data["quantities"]
        """
        if hasattr(self.forward_model, "fit"):
            print("FITTING:"+str(self.name))
            self.forward_model.fit(message_data["quantities"], message_data["target"])
        return self.forward_model

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

        if hasattr(self.forward_model, "transform"):
            transformed_data = self.forward_model.transform(message_data)
        elif hasattr(self.forward_model, "predict"):
            transformed_data = self.forward_model.predict(message_data)
        else:
            transformed_data = self.forward_model(message_data)

        return transformed_data

class ML_InverseTransformAgent(ML_BaseAgent):
    def _transform(self, message_data):
        """
        Internal function. Inverse transforms and returns message_data["quantities"] using self.model
        """
        if hasattr(self.forward_model, "inverse_transform"):
            transformed_data = self.forward_model.inverse_transform(message_data)

        return transformed_data

class ML_TransformPipelineAgent(ML_TransformAgent):
    """
    Transformer which applies sklearn pipeline style, by combining multiple models in a serial way.

    Need to pass 3 lists: models, superviseds and model params corresponding to the models
    """

    stylesheet = "ellipse"

    def init_parameters(self, pipeline_models=[MLPClassifier],
                        pipeline_params=[], random_state=123,
                        predict_train=False, send_train_model=False, use_dmm=False):
        """
        Initialise model parameters.
        Accepts either a class or a function as model.
        Stores the instantiated model in self.model.

        Parameters
        ----------
        model : class or function

        **pipeline_params : keywords for model parameters instantiation.
        """
        self.pipeline_models = pipeline_models
        self.pipeline_params = pipeline_params

        super(ML_TransformPipelineAgent, self).init_parameters(model=None,
                                                               random_state=random_state,
                                                               predict_train=predict_train,
                                                               send_train_model=False,
                                                               use_dmm=True)

        # assume it as a class model to be initialised
        if pipeline_models is not None:
            self.forward_model = self.instantiate_model()

    def instantiate_model(self):
        new_model = make_pipeline(*[model(**model_param) for model,
                                                             model_param in zip(self.pipeline_models, self.pipeline_params)])
        return new_model

class ML_EvaluateAgent(ML_BaseAgent):
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
            y_pred = message["data"]["quantities"]
            y_true = message["data"]["target"]


            # check if it is a dictionary
            if isinstance(y_pred, dict):
                results = {}
                for y_key in y_pred.keys():
                    temp_results = {evaluate_method.__name__+"-"+y_key:evaluate_method(y_true[y_key], y_pred[y_key], **evaluate_param)
                               for evaluate_method,evaluate_param in zip(self.evaluate_methods,self.evaluate_params)}
                    results.update(temp_results)


            else:
                results = {evaluate_method.__name__:evaluate_method(y_true, y_pred, **evaluate_param)
                           for evaluate_method,evaluate_param in zip(self.evaluate_methods,self.evaluate_params)}
            self.log_info(str(results))
            self.upload_result(results)

            # regression
            if self.evaluate_method == "rmse":
                if isinstance(y_pred, dict):
                    graph_comparison = [self.plot_comparison(y_true_i, y_pred_i,
                                                            from_agent=message['from'] + "("+key+")",
                                                            sum_performance="RMSE: " + str(result))
                                        for y_true_i,y_pred_i, key, result in zip(y_true.values(),y_pred.values(),results.keys(), results.values())]
                else:
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


class ML_AggregatorAgent(ML_BaseAgent):
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

class ML_PlottingAgent(ML_TransformAgent):

    def on_received_message(self, message):
        """
        Handles data from fit/test/simulate channels
        """
        # depending on the channel, we train/test using the message's content.
        channel = message["channel"]

        self.handle_channels(message, channel)

        if (channel in ["train","test","simulate"]):
            plot = self.dmm_wrap(self.plot_data, message_data=message["data"], metadata=self.metadata, additional_id=channel+message["from"])
            self.buffer.store(agent_from=message["from"], data=plot)
            self.send_plot(self.get_buffered_plots())

    def get_buffered_plots(self):
        plots = [plot for plot in self.buffer.values()]
        plots = list(np.concatenate(plots, axis=0))

        return plots

    def plot_data(self, message_data, metadata, **kwargs):
        x = message_data["quantities"]
        y = message_data["target"]
        plot = self.forward_model(x, y, metadata,  **kwargs)
        return plot

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
