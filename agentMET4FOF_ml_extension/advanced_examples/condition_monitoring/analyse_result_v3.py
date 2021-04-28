# plots the graph and table from the MLEXP runs for publication purposes

import pandas as pd
import pickle
import matplotlib.pyplot as plt

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
# pd.set_option('display.max_rows', 900000)

def load_bae_results(filename = "MLEXP-Explainability_old/unsupervised-BAE.p"):
    with open(filename,"rb") as f:
        results_df = pickle.load(f)

        cleaned_df = results_df.astype(str).drop(["Date"],axis=1).drop_duplicates(keep="last")
        cleaned_df["total_model_cap"] = cleaned_df["model_capacity"]+"+"+cleaned_df["n_layers"]
        # cleaned_df["total_model_cap"] = cleaned_df["n_layers"]+"+"+cleaned_df["model_capacity"]
        cleaned_df["perf_score"] = cleaned_df["perf_score"].astype(float)

        cleaned_df.loc[cleaned_df["perf_name"].str.contains("var"),"mean-var"] = "var"
        cleaned_df.loc[~cleaned_df["perf_name"].str.contains("var"),"mean-var"] = "mean"

        cleaned_df["perf_name"] = cleaned_df["perf_name"].str.split("_").str[-1]
        cleaned_df["model_type"] = cleaned_df["bae_or_ae"] + "-" \
                                   + cleaned_df["bae_config"] + "-" \
                                   + cleaned_df["explanation"] + "-" \
                                   + cleaned_df["mean-var"]
        cleaned_df["perf_name"].replace("pearson-var", "pearson", inplace=True)
        cleaned_df["perf_name"].replace("pearson-mean", "pearson", inplace=True)

        cleaned_df = cleaned_df.reset_index(drop=True)
    return cleaned_df

def prepare_cd_diagram(filename = "MLEXP-Explainability_old/unsupervised-BAE.p", perf_key ="gmean-sdc"):
    if isinstance(filename, str):
        cleaned_df = load_bae_results(filename)
    elif isinstance(filename, pd.DataFrame):
        cleaned_df = filename

    print(cleaned_df.columns)
    # print(cleaned_df.groupby(["dataset","model_type","total_model_cap","perf_name"]).mean())
    # perf_key = "gmean-sdc"
    # filtered_df = cleaned_df[cleaned_df["perf_name"] == perf_key]
    print(cleaned_df.groupby(["dataset","model_type","total_model_cap","perf_name"])["perf_score"].mean())

    cleaned_df = cleaned_df.reset_index(drop=True)
    cleaned_df["condition"] = cleaned_df["perturbed"] + \
                              cleaned_df["axis"] + \
                              cleaned_df["dataset"] + \
                              cleaned_df["total_model_cap"]

    cd_diagram_pd = cleaned_df[cleaned_df["perf_name"]==perf_key][["model_type","condition","perf_score"]]

    return cd_diagram_pd