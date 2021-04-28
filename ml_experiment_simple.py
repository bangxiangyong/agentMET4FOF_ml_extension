import pandas as pd
import os
import pickle
import datetime

from agentMET4FOF_ml_extension.ml_experiment import ML_ExperimentLite


class ML_ExperimentSimple():
    def __init__(self, ml_exp_name = "simple1",
                 ml_parameters = {"train_size":0.7, "dataset":"ZEMA", "perturbed_sensors":[0,1,2,3]},
                 base_folder = "MLEXP/",
                 ):
        self.ml_exp_name = ml_exp_name
        self.ml_parameters = ml_parameters
        self.base_folder = base_folder
        self.file_path = self.base_folder + self.ml_exp_name + ".p"

    def create_results(self, ml_performance:dict, ml_parameters=None):
        ml_results = {"Date":datetime.datetime.now()}
        ml_results.update(ml_parameters)
        ml_results.update(ml_performance)
        ml_results = pd.DataFrame([ml_results])
        return ml_results

    def load_results(self):
        # create folder
        if not os.path.exists(self.base_folder):
            os.mkdir(self.base_folder)

        # load pickle
        ml_results = pickle.load(open(self.base_folder + self.ml_exp_name + ".p", "rb"))
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
            current_ml_results = pickle.load(open(self.file_path, "rb"))
            if self.check_entry_exist(current_ml_results, new_ml_results, cols=list(ml_parameters.keys())):
                print("Entry exists...")
                return 0
            else:
                new_ml_results = current_ml_results.append(new_ml_results)

        # save pickle
        pickle.dump(new_ml_results, open(self.file_path,"wb"))
        print("Saving results...")

    def check_entry_exist(self, main_df, sub_df, cols=[],drop_date=True):
        if len(cols) == 0 :
            cols = main_df.columns

        if drop_date and "Date" in cols:
            exists = (main_df[cols].drop(labels=["Date"], axis=1) == sub_df[cols].drop(labels=["Date"], axis=1).iloc[0]).all(1).any()
        else:
            exists = (main_df[cols] == sub_df[cols].iloc[0]).all(1).any()
        return exists


new_ml_exp = ML_ExperimentLite()
new_ml_exp.save_results(ml_performance={"f1-score":123,"sensitivity":3981})
new_ml_exp.save_results(ml_performance={"f1-score":555,"sensitivity":9879})
