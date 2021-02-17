from sklearn.model_selection import train_test_split

from agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from agentMET4FOF_ml_extension.ml_agents import ML_DatastreamAgent
from agentMET4FOF_ml_extension.datastreams import Liveline_DataStream, ZEMA_DataStream
import numpy as np

# Datastream Agent

class Liveline_DatastreamAgent(ML_DatastreamAgent):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.
    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"input_stage": [1,2], "target_stage": [1,2], "train_size": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

    def init_parameters(self, input_stage=1, target_stage=1, train_size=0.5, simulate_batch_size=200, random_state=123):
        self.input_stage = input_stage
        self.target_stage = target_stage
        self.train_size = train_size
        self.set_random_state(random_state)

        liveline_datastream = Liveline_DataStream(output_stage=target_stage,
                                                  input_stage=input_stage,
                                                  train_size=train_size)
        self.X_columns = liveline_datastream.X_columns

        self.set_data_sources(x_train=liveline_datastream._quantities["train"],
                              x_test=liveline_datastream._quantities["test"],
                              x_ood=liveline_datastream._quantities["ood"],
                              y_train=liveline_datastream._target["train"],
                              y_test=liveline_datastream._target["test"],
                              y_ood=liveline_datastream._target["ood"])

        # for simulation
        self.datastream_simulate = DataStreamMET4FOF()
        self.datastream_simulate.set_data_source(quantities=np.concatenate((self.x_test,self.x_ood)))
        self.simulate_batch_size = simulate_batch_size

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_output({"quantities":self.x_train, "target":self.y_train},channel="train")
            self.send_output({"quantities":{"test":self.x_test,"ood":self.x_ood},
                              "target":{"test":self.y_test,"ood":self.y_ood}},
                             channel="test")
            # self.current_state = "Idle"
            self.current_state = "Simulate"

        elif self.current_state == "Simulate":
            self.send_output({"quantities": self.datastream_simulate.next_sample(batch_size=self.simulate_batch_size)["quantities"],
                              "metadata":self.X_columns},
                             channel="simulate")

class ZEMA_DatastreamAgent(ML_DatastreamAgent):
    parameter_choices = {"axis":[3,5,7], "train_size":[0.5,0.6,0.7,0.8,0.9]}

    def init_parameters(self, train_axis=[3], test_axis=[], train_size=0.8, random_state=123, shuffle=True, move_axis=True, **data_params):
        """
        x_train : numpy array of sensor data
        x_test : list of numpy arrays for each axis specified in test_axis

        """
        self.set_random_state(random_state)
        self.train_size = train_size
        self.shuffle = shuffle

        x_trains = []
        x_tests = {}
        y_trains = []
        y_tests = {}

        # configure training datasets
        for axis in train_axis:
            datastream = ZEMA_DataStream(axis=axis)
            x_train, x_test, y_train, y_test = train_test_split(datastream._quantities, datastream._target,
                                                                train_size=self.train_size, shuffle=shuffle,
                                                                random_state=random_state)
            x_trains.append(x_train)
            y_trains.append(y_train)

            if axis in test_axis:
                x_tests.update({str(axis): x_test})
                y_tests.update({str(axis): y_test})

        # configure test datasets
        for axis in test_axis:
            datastream = ZEMA_DataStream(axis=axis)
            # if the axis has been specified in train axis,
            # it has been splitted in the block of code before this
            # and included in the x_tests
            # hence, we don't need to add it again to the list

            if axis not in train_axis:
                x_tests.update({str(axis): datastream._quantities})
                y_tests.update({str(axis): datastream._target})

        # concatenate into numpy array
        x_trains = np.concatenate(x_trains, axis=0)
        y_trains = np.concatenate(y_trains, axis=0)

        # move axis
        if move_axis:
            x_trains = np.moveaxis(x_trains, 1, 2)
            x_tests = np.moveaxis(x_tests, 1, 2)

        self.set_data_sources(x_train = x_trains,
                              x_test  = x_tests,
                              y_train = y_trains,
                              y_test  = y_tests)

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_output({"quantities":self.x_train, "target":self.y_train},channel="train")
            if len(self.x_test) > 0:
                self.send_output({"quantities":self.x_test,
                                  "target":self.y_test},
                                 channel="test")

            self.current_state = "Idle"