# plots the graph and table from the MLEXP runs for publication purposes

import pandas as pd
import pickle
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 900000)

results_df = pickle.load(open("MLEXP-Explainability/unsupervised-BAE.p","rb"))
# results_df = pickle.load(open("MLEXP_old/unsupervised-BAE.p","rb"))
# results_df = pickle.load(open("MLEXP_old/unsupervised-BAE.p","rb"))

cleaned_df = results_df.astype(str).drop(["Date"],axis=1).drop_duplicates(keep="last")
cleaned_df["total_model_cap"] = cleaned_df["model_capacity"]+"+"+cleaned_df["n_layers"]
cleaned_df["perf_score"] = cleaned_df["perf_score"].astype(float)

cleaned_df.loc[cleaned_df["perf_name"].str.contains("var"),"mean-var"] = "var"
cleaned_df.loc[~cleaned_df["perf_name"].str.contains("var"),"mean-var"] = "mean"

# cleaned_df["perf_name"] = cleaned_df["perf_name"].str.split("_").str[-1]
cleaned_df["model_type"] = cleaned_df["bae_or_ae"] + "-" \
                           + cleaned_df["bae_config"] + "-" \
                           + cleaned_df["explanation"] + "-" \
                           + cleaned_df["mean-var"]

# cleaned_df.loc[cleaned_df["perf_name"].str.contains("var"),"mean-var"] = 'False'


# filter by performance name
def filter_contains(df, cols=["perf_name"], string=["gmean_sdc"]):
    new_df = df.copy()
    for (col, string_i) in zip(cols, string):
        new_df = new_df[new_df[col].str.contains(string_i)]
    return new_df

# perf_key = "gmean-sser"
perf_key = "gmean-sdc"
# perf_key = "gmean_sdc"
# perf_key = "spman-sdc"

# gmean_filtered_mean = filter_contains(cleaned_df, cols=["perf_name","perf_name"], string=[perf_key,"mean"])
# gmean_filtered_var = filter_contains(cleaned_df, cols=["perf_name","perf_name"], string=[perf_key,"var"])

gmean_filtered_mean = filter_contains(cleaned_df, cols=["perf_name"], string=[perf_key])
# gmean_filtered_var = filter_contains(cleaned_df, cols=["perf_name"], string=[perf_key])

# gmean_filtered_mean["method"] = gmean_filtered_mean["model_type"] + "-mean"
# gmean_filtered_var["method"] = gmean_filtered_var["model_type"] + "-var"
# gmean_filtered_mean["perf_name"] = perf_key
# gmean_filtered_var["perf_name"] = perf_key

print(gmean_filtered_mean.groupby(["dataset","model_type","total_model_cap"]).mean())
# print(gmean_filtered_var.groupby(["dataset","model_type","total_model_cap"]).mean())

print(gmean_filtered_mean["perf_score"].mean())

# new_df = gmean_filtered_mean.append(gmean_filtered_var)
# new_df = new_df[["method","total_model_cap","perf_name","perf_score"]]

# pd.set_option('display.max_columns', None)
# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('max_colwidth', -1)
# pd.set_option('display.max_rows', 900000)
# # print(cleaned_df.groupby(["dataset","model_type","total_model_cap","perf_name"]).mean())
# perf_key = "tnr-spman"
# filtered_df = cleaned_df[cleaned_df["perf_name"] == perf_key]
# print(filtered_df.groupby(["dataset","model_type","total_model_cap"])["perf_score"].mean())
