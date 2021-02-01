import pickle
import os
from new_ml_experiment import ML_Results
import pandas as pd
# import matplotlib.pyplot as plt



def load_ml_results(base_folder = "MLEXP/"):
    result_filenames = os.listdir(base_folder)
    ml_results = [pickle.load(open(base_folder + filename, "rb")) for filename in result_filenames]

    return ml_results

def load_filtered_ml_res(run_names=[], dates=[], base_folder = "MLEXP/"):
    ml_results = [pickle.load(open(base_folder + filename+".p", "rb")) for filename in run_names]
    # ml_results = [ml_result for ml_result in ml_results if ml_result.run_details.date[0] in dates]
    return ml_results

def load_pd_full(run_names=[], dates=[], convert_str=True, base_folder = "MLEXP/"):
    if len(run_names) == 0 and len(dates) == 0:
        ml_results = load_ml_results(base_folder)
    else:
        ml_results = load_filtered_ml_res(run_names,dates,base_folder)
    ml_results_pd = pd.concat([ml_result.get_results_pd() for ml_result in ml_results]).reset_index(drop=True)

    if convert_str:
        obj_cols = ["data", "data_params", "model", "model_params"]
        ml_results_pd[obj_cols] = ml_results_pd[obj_cols].astype(str)
    return ml_results_pd

def load_ml_exp_details(base_folder = "MLEXP/"):
    ml_results = load_ml_results(base_folder)
    ml_details = pd.concat([ml_result.run_details for ml_result in ml_results]).reset_index(drop=True)

    return ml_details


def groupby_results(ml_results_pd, groupby_cols = ["data", "data_params", "model", "model_params"], perf_columns=["perf_score"], reset_index=True):
    groupby_cols = groupby_cols+["perf_name"]
    temp_pd = ml_results_pd.copy()
    temp_pd[groupby_cols] = ml_results_pd[groupby_cols].astype(str)

    temp_pd_mean = temp_pd.groupby(groupby_cols).mean()[perf_columns]
    temp_pd_sem = temp_pd.groupby(groupby_cols).std()[perf_columns]

    if reset_index:
        return temp_pd_mean.reset_index(), temp_pd_sem.reset_index()
    else:
        return temp_pd_mean, temp_pd_sem

ml_results = load_ml_results(base_folder = "MLEXP/")
ml_exp_details = load_ml_exp_details()


ml_results_pd = load_pd_full(base_folder = "MLEXP/")

pd_mean, pd_sem= groupby_results(ml_results_pd, perf_columns=["perf_score"])

# pd.DataFrame(ml_run.results)



# ml_filtered= filter_results(run_names=[ml_exp_details.run_name[0]], dates=[ml_exp_details.date[0]], base_folder = "MLEXP/")








