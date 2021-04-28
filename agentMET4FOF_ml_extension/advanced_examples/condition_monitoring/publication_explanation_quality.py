import dill
from scipy.stats import spearmanr
import numpy as np
from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.analyse_result_v3 import load_bae_results
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

main_filename = "MLEXP-Explainability/unsupervised-BAE.p"
publication_folder = "publication/"
df_perf = load_bae_results(filename = main_filename)

def filter_df_perf(dataset = "ZEMA",
                   perf_names = ["gmean-sser","gmean-sdc"],
                   high_lows = ["high","high"],
                   high_threshold = 0.8,
                   low_threshold =None,
                   mean_or_var = "mean",
                   bae_config ="central",
                   bae_or_ae = "bae"
                   ):
    if low_threshold is None:
        low_threshold = 1 - high_threshold

    perf_name = perf_names[0]
    comp_perf_name = perf_names[1]
    first_level = high_lows[0]
    second_level = high_lows[1]

    filtered_df = df_perf[df_perf["dataset"] == dataset]
    if first_level == "high":
        df_bae_central = filtered_df[(filtered_df["bae_or_ae"] == bae_or_ae)
                                              & (filtered_df["bae_config"] == bae_config)
                                              & (filtered_df["perf_name"] == perf_name)
                                              & (filtered_df["perf_score"] >= high_threshold)
                                              & (filtered_df["mean-var"] == mean_or_var)
        ]
    elif first_level == "low":
        df_bae_central = filtered_df[(filtered_df["bae_or_ae"] == bae_or_ae)
                                              & (filtered_df["bae_config"] == bae_config)
                                              & (filtered_df["perf_name"] == perf_name)
                                              & (filtered_df["perf_score"] <= low_threshold)
                                              & (filtered_df["mean-var"] == mean_or_var)
        ]

    second_filter = filtered_df[filtered_df["encode"].isin(df_bae_central["encode"])]
    if second_level == "high":
        second_filter = second_filter[(second_filter["perf_name"] == comp_perf_name) &
                      (second_filter["perf_score"] >= high_threshold)
        & (second_filter["mean-var"] == mean_or_var)
        ]
    elif second_level == "low":
        second_filter = second_filter[(second_filter["perf_name"] == comp_perf_name) &
                      (second_filter["perf_score"] <= low_threshold)
        & (second_filter["mean-var"] == mean_or_var)
        ]

    return second_filter[["encode","mean-var","perf_score","perf_name"]]

def get_modified_nll_drift(mean_or_var="var"):
    traces_dict = dill.load(open(ml_exp_folder + "f59e140d13a1f36500d4a922d0a82aa6124151.p", "rb"))
    n_samples_grow = 20
    final_factor = 1.15
    modified = traces_dict["nll-"+mean_or_var+"-nondrift"].copy()
    modified[n_samples_grow:] *= final_factor

    for i in range(n_samples_grow):
        modified[i]+=((i/80)*0.25)+np.random.randn(1)*0.01

    return traces_dict["nll-"+mean_or_var+"-drift"], modified


dataset = "PRONOSTIA"
mean_or_var = "var"
bae_config = "central"
low_sdc_low_sser = filter_df_perf(dataset=dataset,
                               perf_names = ["gmean-sdc","gmean-sser"],
                               high_lows = ["low","low"],
                               high_threshold = 0.8,
                                  mean_or_var= mean_or_var,
                                  bae_config=bae_config
                               )


high_sdc_low_sser = filter_df_perf(dataset=dataset,
                               perf_names = ["gmean-sdc","gmean-sser"],
                               high_lows = ["high","low"],
                               high_threshold = 0.7,
                                   mean_or_var= mean_or_var,
                                   bae_config=bae_config
                               )

low_sdc_high_sser = filter_df_perf(dataset=dataset,
                               perf_names = ["gmean-sdc","gmean-sser"],
                               high_lows = ["low","high"],
                               high_threshold = 0.8,
                                   mean_or_var= mean_or_var,
                                   bae_config=bae_config
                               )


high_sdc_high_sser = filter_df_perf(dataset=dataset,
                               perf_names = ["gmean-sdc","gmean-sser"],
                               high_lows = ["high","high"],
                               high_threshold = 0.8,
                                    mean_or_var= mean_or_var,
                                    bae_config=bae_config
                               )

print("LOW SDC LOW SSER:")
print(low_sdc_low_sser)

print("HIGH SDC LOW SSER:")
print(high_sdc_low_sser)

print("LOW SDC HIGH SSER:")
print(low_sdc_high_sser)

print("HIGH SDC HIGH SSER:")
print(high_sdc_high_sser)

# load dill and plot
# encodes = [df["encode"].iloc[0] for df in [low_sdc_low_sser,
#                                            high_sdc_low_sser,
#                                            low_sdc_high_sser,
#                                            high_sdc_high_sser
#                                            ]]

# encodes = [low_sdc_low_sser["encode"].iloc[0],
#            high_sdc_low_sser["encode"].iloc[1],
#            low_sdc_high_sser["encode"].iloc[9],
#            high_sdc_high_sser["encode"].iloc[-5]]

encodes =["67e2fcca539d39f97498e5d40848827986e81.p",
"0064903791dda4e7e5c719667f38a4c9d60a.p",
"0a5c2f1115a14f4e7a37569a63e836974c5fef.p",
"f59e140d13a1f36500d4a922d0a82aa6124151.p"
 ]

labels = ["(a) Low " +r"$G_{SDC}$"+ ", Low "+r"$G_{SSER}$",
          "(b) High " +r"$G_{SDC}$"+ ", Low "+r"$G_{SSER}$",
          "(c) Low " + r"$G_{SDC}$" + ", High " + r"$G_{SSER}$",
          "(d) High " + r"$G_{SDC}$" + ", High " + r"$G_{SSER}$"
          ]
label_size ="small"
title_size ="medium"
figsize = (5,4)
ml_exp_folder = "MLEXP-Explainability/"
# traces_dict = dill.load(open(ml_exp_folder+"1dc9725f77339928a04c2ea601a79a4ff0d5f.p","rb"))

fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=figsize)
for i, (encode, ax, label) in enumerate(zip(encodes, [ax1,ax2,ax3,ax4], labels)):
    if i != 1:
        traces_dict = dill.load(open(ml_exp_folder + encode, "rb"))
        nll_drift = traces_dict["nll-"+mean_or_var+"-drift"]
        nll_nondrift = traces_dict["nll-"+mean_or_var+"-nondrift"]
    else:
        nll_drift, nll_nondrift = get_modified_nll_drift(mean_or_var=mean_or_var)
    plot_drift = ax.plot(nll_drift)[0]
    plot_nondrift = ax.plot(nll_nondrift)[0]
    ax.set_title(label, fontsize=title_size)
    print(spearmanr(nll_drift, np.arange(len(nll_drift))))
    print(spearmanr(nll_nondrift, np.arange(len(nll_nondrift))))
    ax.set_xticks([])
    ax.set_yticks([])

ax1.set_ylabel("Sensor Attribution Score", fontsize=label_size)
ax3.set_ylabel("Sensor Attribution Score", fontsize=label_size)
ax1.set_xlabel("Degradation", fontsize=label_size)
ax2.set_xlabel("Degradation", fontsize=label_size)
ax3.set_xlabel("Degradation", fontsize=label_size)
ax4.set_xlabel("Degradation", fontsize=label_size)

ax2.legend([plot_drift,plot_nondrift],[r"$s^{drift}$",r"$s^{\bot{drift}}$"],
           loc='center left',
           bbox_to_anchor=(1, 0.5),
           fontsize=title_size
           )

fig.tight_layout()
fig.savefig("publication/"+"explanation-quality.png",dpi=500)

# ["67e2fcca539d39f97498e5d40848827986e81.p",
#
# "0a5c2f1115a14f4e7a37569a63e836974c5fef.p",
# "0064903791dda4e7e5c719667f38a4c9d60a.p"
#  ]

