from sklearn.model_selection import train_test_split

from agentMET4FOF.agentMET4FOF.streams import DataStreamMET4FOF
from agentMET4FOF_ml_extension.ml_agents import ML_DatastreamAgent
from agentMET4FOF_ml_extension.datastreams import Liveline_DataStream, ZEMA_DataStream, STRATH_Datastream, \
    PRONOSTIA_DataStream
import numpy as np
import pandas as pd

# Datastream Agent

class Liveline_DatastreamAgent(ML_DatastreamAgent):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.
    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"input_stage": [1, 2], "target_stage": [1, 2],
                         "train_size": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    def init_parameters(self, input_stage=1, target_stage=1, train_size=0.5, simulate_batch_size=200, random_state=123,
                        use_dmm=True):
        super(Liveline_DatastreamAgent, self).init_parameters(datastream=None,
                                                              train_size=train_size,
                                                              random_state=random_state,
                                                              use_dmm=use_dmm)

        self.input_stage = input_stage
        self.target_stage = target_stage

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
        self.datastream_simulate.set_data_source(quantities=np.concatenate((self.x_test, self.x_ood)))
        self.simulate_batch_size = simulate_batch_size

    def agent_loop(self):
        if self.current_state == "Running":
            if self.use_dmm:
                self.send_dmm_code()

            self.send_output({"quantities": self.x_train, "target": self.y_train}, channel="train")
            self.send_output({"quantities": {"test": self.x_test, "ood": self.x_ood},
                              "target": {"test": self.y_test, "ood": self.y_ood}},
                             channel="test")
            # self.current_state = "Idle"
            self.current_state = "Simulate"

        elif self.current_state == "Simulate":
            self.send_output(
                {"quantities": self.datastream_simulate.next_sample(batch_size=self.simulate_batch_size)["quantities"],
                 "metadata": self.X_columns},
                channel="simulate")


class STRATH_DatastreamAgent(ML_DatastreamAgent):
    """
    A base class for ML data-streaming agent, which takes into account the train/test split.
    The agent_loop behaviour are based on "current_state":
    1) in "Train", it will send out train and test batch of data, and then set to "Idle" state
    2) in "Simulate", it will send out test data one-by-one in a datastream fashion.
    3) Output form : {"quantities": iterable, "target": iterable}
    """

    parameter_choices = {"input_stage": [1, 2], "target_stage": [1, 2],
                         "train_size": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}

    def init_parameters(self, phase="heating", train_size=0.5, shuffle=True, simulate_batch_size=200, random_state=123,
                        use_dmm=True):
        super(STRATH_DatastreamAgent, self).init_parameters(datastream=None,
                                                            train_size=train_size,
                                                            random_state=random_state,
                                                            use_dmm=use_dmm)

        datastream = STRATH_Datastream(phase=phase)
        self.X_columns = datastream.xlabels
        self.shuffle = shuffle
        x_ood = datastream._quantities[datastream.arg_outliers]
        y_ood = datastream._target[datastream.arg_outliers]

        x_train, x_test, y_train, y_test = train_test_split(datastream._quantities[datastream.arg_inliers],
                                                            datastream._target[datastream.arg_inliers],
                                                            train_size=self.train_size,
                                                            shuffle=self.shuffle,
                                                            random_state=self.random_state)

        self.set_data_sources(x_train=x_train,
                              x_test=x_test,
                              x_ood=x_ood,
                              y_train=y_train,
                              y_test=y_test,
                              y_ood=y_ood)

        # for simulation
        self.datastream_simulate = DataStreamMET4FOF()
        self.datastream_simulate.set_data_source(quantities=np.concatenate((self.x_test, self.x_ood)))
        self.simulate_batch_size = simulate_batch_size

    def agent_loop(self):
        if self.current_state == "Running":
            self.send_dmm_code()

            self.send_output({"quantities": self.x_train, "target": self.y_train}, channel="train")
            self.send_output({"quantities": {"test": self.x_test, "ood": self.x_ood},
                              "target": {"test": self.y_test, "ood": self.y_ood}},
                             channel="test")
            self.current_state = "Idle"
            # self.current_state = "Simulate"

        elif self.current_state == "Simulate":
            self.send_output(
                {"quantities": self.datastream_simulate.next_sample(batch_size=self.simulate_batch_size)["quantities"],
                 "metadata": self.X_columns},
                channel="simulate")


class ZEMA_DatastreamAgent(ML_DatastreamAgent):
    parameter_choices = {"axis": [3, 5, 7], "train_size": [0.5, 0.6, 0.7, 0.8, 0.9]}

    def init_parameters(self, train_axis=[3], test_axis=[], train_size=0.8, random_state=123,
                        use_dmm=True, shuffle=True, send_test_samples=False, return_full_test=False,
                        cut_first_perc=0, cut_last_perc=0, datastream_class=ZEMA_DataStream,
                        **data_params):

        """
        x_train : numpy array of sensor data
        x_test : list of numpy arrays for each axis specified in test_axis
        """
        self.train_axis = train_axis
        self.test_axis = test_axis
        self.shuffle = shuffle
        self.send_test_samples = send_test_samples
        self.return_full_test = return_full_test
        self.cut_first_perc = cut_first_perc
        self.cut_last_perc = cut_last_perc
        self.datastream_class = datastream_class

        super(ZEMA_DatastreamAgent, self).init_parameters(datastream=None,
                                                          train_size=train_size,
                                                          random_state=random_state,
                                                          use_dmm=use_dmm)

        x_trains, x_tests, y_trains, y_tests, metadata = self.dmm_wrap(self.prepare_data)
        self.metadata = metadata
        self.set_data_sources(x_train=x_trains,
                              x_test=x_tests,
                              y_train=y_trains,
                              y_test=y_tests)

    def prepare_data(self):
        x_trains = []
        x_tests = {}
        y_trains = []
        y_tests = {}

        # configure training datasets
        for axis in self.train_axis:
            datastream = self.datastream_class(axis=axis)

            datastream._quantities, datastream._target = self.truncate_data_xy(datastream._quantities,
                                                                               datastream._target)

            x_train, x_test, y_train, y_test = train_test_split(datastream._quantities, datastream._target,
                                                                train_size=self.train_size, shuffle=self.shuffle,
                                                                random_state=self.random_state)
            x_trains.append(x_train)
            y_trains.append(y_train)

            if axis in self.test_axis:
                if not self.return_full_test:
                    x_tests.update({str(axis): x_test})
                    y_tests.update({str(axis): y_test})
                else:
                    x_tests.update({str(axis): datastream._quantities})
                    y_tests.update({str(axis): datastream._target})

        # configure test datasets
        for axis in self.test_axis:
            datastream = self.datastream_class(axis=axis)
            # if the axis has been specified in train axis,
            # it has been splitted in the block of code before this
            # and included in the x_tests
            # hence, we don't need to add it again to the list

            if axis not in self.train_axis:
                x_tests.update({str(axis): datastream._quantities})
                y_tests.update({str(axis): datastream._target})

        # get metadata
        metadata = {"units": datastream.units, "labels": datastream.labels}

        # concatenate into numpy array
        x_trains = np.concatenate(x_trains, axis=0)
        y_trains = np.concatenate(y_trains, axis=0)
        return x_trains, x_tests, y_trains, y_tests, metadata

    def agent_loop(self):
        if self.current_state == "Running":
            # send dmm code
            self.send_dmm_code()

            # send metadata
            if hasattr(self, "metadata"):
                self.send_output(self.metadata, channel="metadata")

            # send training data
            self.send_output({"quantities": self.x_train, "target": self.y_train}, channel="train")

            # send test data
            if len(self.x_test) > 0:
                self.send_test_data()

                # if send_test_samples is enabled, we send some random test_samples to be plotted
                if self.send_test_samples:
                    x_random, y_random = self.get_random_examples(self.x_test, self.y_test)
                    self.send_output({"quantities": x_random,
                                      "target": y_random,
                                      },
                                     channel="test_samples")

            self.current_state = "Idle"

    def send_test_data(self):
        self.send_output({"quantities": self.x_test,
                          "target": self.y_test,
                          },
                         channel="test")

    def send_train_data(self):
        self.send_output({"quantities": self.x_train, "target": self.y_train}, channel="train")


class PRONOSTIA_DatastreamAgent(ZEMA_DatastreamAgent):
    parameter_choices = {"axis": ["1_1",
                                  "1_2",
                                  "1_3"
                                  "1_4",
                                  "1_5",
                                  "1_6",
                                  "1_7",
                                  "2_1",
                                  "2_2",
                                  "2_3"
                                  "2_4",
                                  "2_5",
                                  "2_6",
                                  "2_7",
                                  "3_1",
                                  "3_2",
                                  "3_3"
                                  ], "train_size": [0.5, 0.6, 0.7, 0.8, 0.9]}

    def init_parameters(self, train_axis=[3], test_axis=[], train_size=0.8, random_state=123,
                        use_dmm=True, shuffle=True, send_test_samples=False, return_full_test=False,
                        cut_first_perc=0, cut_last_perc=0, datastream_class=ZEMA_DataStream,
                        **data_params):

        super(PRONOSTIA_DatastreamAgent, self).init_parameters(
            train_axis=train_axis, test_axis=test_axis, train_size=train_size,
            random_state=random_state,
            use_dmm=use_dmm, shuffle=shuffle, send_test_samples=send_test_samples,
            return_full_test=return_full_test,
            cut_first_perc=cut_first_perc, cut_last_perc=cut_last_perc, datastream_class=PRONOSTIA_DataStream,
            **data_params)
