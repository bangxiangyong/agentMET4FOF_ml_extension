from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.analyse_result_v3 import load_bae_results
import pandas as pd


main_filename = "MLEXP-Explainability/unsupervised-BAE.p"
publication_folder = "publication/"
df_perf = load_bae_results(filename = main_filename)


# parameters
dataset = "ZEMA"
threshold_pearson = 0.9

# start filter dfs
filtered_df = df_perf[df_perf["dataset"] == dataset]
df_bae_central = filtered_df[(filtered_df["bae_or_ae"] == "bae")
                                      & (filtered_df["bae_config"] == "central")
                                      & (filtered_df["perf_name"] == "pearson")
                                      & (filtered_df["perf_score"] > threshold_pearson)
]



df_bae_coalition = filtered_df[(filtered_df["bae_or_ae"] == "bae")
                                      & (filtered_df["bae_config"] == "coalition")
                                      & (filtered_df["perf_name"] == "pearson")
]

select_row_central = df_bae_central.iloc[0]
select_row_coalition = df_bae_coalition[
    (df_bae_coalition["total_model_cap"] == select_row_central["total_model_cap"]) &
    (df_bae_coalition["perturbed"] == select_row_central["perturbed"]) &
    (df_bae_coalition["axis"] == select_row_central["axis"])
]
print(select_row_central["encode"])
print(select_row_coalition["encode"].unique()[0])

pairs = {"central":[],"coalition":[]}
for encode in list(df_bae_central["encode"].unique()):
    select_row_central = df_bae_central[df_bae_central["encode"]==encode].iloc[0]
    select_row_coalition = df_bae_coalition[
        (df_bae_coalition["total_model_cap"] == select_row_central["total_model_cap"]) &
        (df_bae_coalition["perturbed"] == select_row_central["perturbed"]) &
        (df_bae_coalition["axis"] == select_row_central["axis"])
    ].iloc[0]
    pairs["central"].append(select_row_central["encode"])
    pairs["coalition"].append(select_row_coalition["encode"])
pairs = pd.DataFrame(pairs)
pd.set_option('display.max_rows', 500)
print(pairs)



















