# Author: Hassan Ismail Fawaz <hassan.ismail-fawaz@uha.fr>
#         Germain Forestier <germain.forestier@uha.fr>
#         Jonathan Weber <jonathan.weber@uha.fr>
#         Lhassane Idoumghar <lhassane.idoumghar@uha.fr>
#         Pierre-Alain Muller <pierre-alain.muller@uha.fr>
# License: GPL3

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from agentMET4FOF_ml_extension.advanced_examples.condition_monitoring.analyse_result_v3 import prepare_cd_diagram, \
    load_bae_results

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

import operator
import math
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare
import networkx

# inspired from orange3 https://docs.orange.biolab.si/3/data-mining-library/reference/evaluation.cd.html
def graph_ranks(avranks, names, p_values, cd=None, cdmethod=None, lowv=None, highv=None,
                width=6, textspace=1, reverse=False, filename=None, labels=False, **kwargs):
    """
    Draws a CD graph, which is used to display  the differences in methods'
    performance. See Janez Demsar, Statistical Comparisons of Classifiers over
    Multiple Data Sets, 7(Jan):1--30, 2006.

    Needs matplotlib to work.

    The image is ploted on `plt` imported using
    `import matplotlib.pyplot as plt`.

    Args:
        avranks (list of float): average ranks of methods.
        names (list of str): names of methods.
        cd (float): Critical difference used for statistically significance of
            difference between methods.
        cdmethod (int, optional): the method that is compared with other methods
            If omitted, show pairwise comparison of methods
        lowv (int, optional): the lowest shown rank
        highv (int, optional): the highest shown rank
        width (int, optional): default width in inches (default: 6)
        textspace (int, optional): space on figure sides (in inches) for the
            method names (default: 1)
        reverse (bool, optional):  if set to `True`, the lowest rank is on the
            right (default: `False`)
        filename (str, optional): output file name (with extension). If not
            given, the function does not write a file.
        labels (bool, optional): if set to `True`, the calculated avg rank
        values will be displayed
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
    except ImportError:
        raise ImportError("Function graph_ranks requires matplotlib.")

    width = float(width)
    textspace = float(textspace)

    def nth(l, n):
        """
        Returns only nth elemnt in a list.
        """
        n = lloc(l, n)
        return [a[n] for a in l]

    def lloc(l, n):
        """
        List location in list of list structure.
        Enable the use of negative locations:
        -1 is the last element, -2 second last...
        """
        if n < 0:
            return len(l[0]) + n
        else:
            return n

    def mxrange(lr):
        """
        Multiple xranges. Can be used to traverse matrices.
        This function is very slow due to unknown number of
        parameters.

        >>> mxrange([3,5])
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        >>> mxrange([[3,5,1],[9,0,-3]])
        [(3, 9), (3, 6), (3, 3), (4, 9), (4, 6), (4, 3)]

        """
        if not len(lr):
            yield ()
        else:
            # it can work with single numbers
            index = lr[0]
            if isinstance(index, int):
                index = [index]
            for a in range(*index):
                for b in mxrange(lr[1:]):
                    yield tuple([a] + list(b))

    def print_figure(fig, *args, **kwargs):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(*args, **kwargs)

    sums = avranks

    nnames = names
    ssums = sums

    if lowv is None:
        lowv = min(1, int(math.floor(min(ssums))))
    if highv is None:
        highv = max(len(avranks), int(math.ceil(max(ssums))))

    cline = 0.4

    k = len(sums)

    lines = None

    linesblank = 0
    scalewidth = width - 2 * textspace

    def rankpos(rank):
        if not reverse:
            a = rank - lowv
        else:
            a = highv - rank
        return textspace + scalewidth / (highv - lowv) * a

    distanceh = 0.25

    cline += distanceh

    # calculate height needed height of an image
    minnotsignificant = max(2 * 0.2, linesblank)
    height = cline + ((k + 1) / 2) * 0.2 + minnotsignificant

    fig = plt.figure(figsize=(width, height))
    fig.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1])  # reverse y axis
    ax.set_axis_off()

    hf = 1. / height  # height factor
    wf = 1. / width

    def hfl(l):
        return [a * hf for a in l]

    def wfl(l):
        return [a * wf for a in l]

    # Upper left corner is (0,0).
    ax.plot([0, 1], [0, 1], c="w")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    def line(l, color='k', **kwargs):
        """
        Input is a list of pairs of points.
        """
        ax.plot(wfl(nth(l, 0)), hfl(nth(l, 1)), color=color, **kwargs)

    def text(x, y, s, *args, **kwargs):
        ax.text(wf * x, hf * y, s, *args, **kwargs)

    line([(textspace, cline), (width - textspace, cline)], linewidth=2)

    bigtick = 0.3
    smalltick = 0.15
    linewidth = 2.0
    linewidth_sign = 4.0

    tick = None
    for a in list(np.arange(lowv, highv, 0.5)) + [highv]:
        tick = smalltick
        if a == int(a):
            tick = bigtick
        line([(rankpos(a), cline - tick / 2),
              (rankpos(a), cline)],
             linewidth=2)

    for a in range(lowv, highv + 1):
        text(rankpos(a), cline - tick / 2 - 0.05, str(a),
             ha="center", va="bottom", size=16)

    k = len(ssums)

    def filter_names(name):
        return name

    space_between_names = 0.24

    for i in range(math.ceil(k / 2)):
        chei = cline + minnotsignificant + i * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace - 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="right", va="center", size=10)
        text(textspace - 0.2, chei, filter_names(nnames[i]), ha="right", va="center", size=16)

    for i in range(math.ceil(k / 2), k):
        chei = cline + minnotsignificant + (k - i - 1) * space_between_names
        line([(rankpos(ssums[i]), cline),
              (rankpos(ssums[i]), chei),
              (textspace + scalewidth + 0.1, chei)],
             linewidth=linewidth)
        if labels:
            text(textspace + scalewidth - 0.3, chei - 0.075, format(ssums[i], '.2f'), ha="left", va="center", size=10)
        text(textspace + scalewidth + 0.2, chei, filter_names(nnames[i]),
             ha="left", va="center", size=16)

    # no-significance lines
    def draw_lines(lines, side=0.05, height=0.1):
        start = cline + 0.2

        for l, r in lines:
            line([(rankpos(ssums[l]) - side, start),
                  (rankpos(ssums[r]) + side, start)],
                 linewidth=linewidth_sign)
            start += height
            print('drawing: ', l, r)

    # draw_lines(lines)
    start = cline + 0.2
    side = -0.02
    height = 0.1

    # draw no significant lines
    # get the cliques
    cliques = form_cliques(p_values, nnames)
    i = 1
    achieved_half = False
    print(nnames)
    for clq in cliques:
        if len(clq) == 1:
            continue
        print(clq)
        min_idx = np.array(clq).min()
        max_idx = np.array(clq).max()
        if min_idx >= len(nnames) / 2 and achieved_half == False:
            start = cline + 0.25
            achieved_half = True
        line([(rankpos(ssums[min_idx]) - side, start),
              (rankpos(ssums[max_idx]) + side, start)],
             linewidth=linewidth_sign)
        start += height

    return fig

def form_cliques(p_values, nnames):
    """
    This method forms the cliques
    """
    # first form the numpy matrix data
    m = len(nnames)
    g_data = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if p[3] == False:
            i = np.where(nnames == p[0])[0][0]
            j = np.where(nnames == p[1])[0][0]
            min_i = min(i, j)
            max_j = max(i, j)
            g_data[min_i, max_j] = 1

    g = networkx.Graph(g_data)
    return networkx.find_cliques(g)


def draw_cd_diagram(df_perf=None, alpha=0.05, title=None, labels=False, file_name='cd-diagram'):
    """
    Draws the critical difference diagram given the list of pairwise classifiers that are
    significant or not
    """
    p_values, average_ranks, _ = wilcoxon_holm(df_perf=df_perf, alpha=alpha)

    print(average_ranks)

    for p in p_values:
        print(p)


    fig = graph_ranks(average_ranks.values, average_ranks.keys(), p_values,
                cd=None, reverse=True, width=9, textspace=1.5, labels=labels)

    font = {'family': 'sans-serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
    if title:
        plt.title(title,fontdict=font, y=0.9, x=0.5)
    # plt.savefig(file_name+'.png',bbox_inches='tight')

    return fig

def wilcoxon_holm(alpha=0.05, df_perf=None):
    """
    Applies the wilcoxon signed rank test between each pair of algorithm and then use Holm
    to reject the null's hypothesis
    """
    print(pd.unique(df_perf['classifier_name']))
    # count the number of tested datasets per classifier
    df_counts = pd.DataFrame({'count': df_perf.groupby(
        ['classifier_name']).size()}).reset_index()
    # get the maximum number of tested datasets
    max_nb_datasets = df_counts['count'].max()
    # get the list of classifiers who have been tested on nb_max_datasets
    classifiers = list(df_counts.loc[df_counts['count'] == max_nb_datasets]
                       ['classifier_name'])
    # test the null hypothesis using friedman before doing a post-hoc analysis
    friedman_p_value = friedmanchisquare(*(
        np.array(df_perf.loc[df_perf['classifier_name'] == c]['accuracy'])
        for c in classifiers))[1]
    if friedman_p_value >= alpha:
        # then the null hypothesis over the entire classifiers cannot be rejected
        print('the null hypothesis over the entire classifiers cannot be rejected')
        exit()
    # get the number of classifiers
    m = len(classifiers)
    # init array that contains the p-values calculated by the Wilcoxon signed rank test
    p_values = []
    # loop through the algorithms to compare pairwise
    for i in range(m - 1):
        # get the name of classifier one
        classifier_1 = classifiers[i]
        # get the performance of classifier one
        perf_1 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_1]['accuracy']
                          , dtype=np.float64)
        for j in range(i + 1, m):
            # get the name of the second classifier
            classifier_2 = classifiers[j]
            # get the performance of classifier one
            perf_2 = np.array(df_perf.loc[df_perf['classifier_name'] == classifier_2]
                              ['accuracy'], dtype=np.float64)
            # calculate the p_value
            p_value = wilcoxon(perf_1, perf_2, zero_method='pratt')[1]
            # appen to the list
            p_values.append((classifier_1, classifier_2, p_value, False))
    # get the number of hypothesis
    k = len(p_values)
    # sort the list in acsending manner of p-value
    p_values.sort(key=operator.itemgetter(2))

    # loop through the hypothesis
    for i in range(k):
        # correct alpha with holm
        new_alpha = float(alpha / (k - i))
        # test if significant after holm's correction of alpha
        if p_values[i][2] <= new_alpha:
            p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)
        else:
            # stop
            break
    # compute the average ranks to be returned (useful for drawing the cd diagram)
    # sort the dataframe of performances
    sorted_df_perf = df_perf.loc[df_perf['classifier_name'].isin(classifiers)]. \
        sort_values(['classifier_name', 'dataset_name'])
    # get the rank data
    rank_data = np.array(sorted_df_perf['accuracy']).reshape(m, max_nb_datasets)

    # create the data frame containg the accuracies
    df_ranks = pd.DataFrame(data=rank_data, index=np.sort(classifiers), columns=
    np.unique(sorted_df_perf['dataset_name']))

    # number of wins
    dfff = df_ranks.rank(ascending=False)
    print(dfff[dfff == 1.0].sum(axis=1))

    # average the ranks
    average_ranks = df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)
    # return the p-values and the average ranks
    return p_values, average_ranks, max_nb_datasets

def replace_full_name_df(df_perf):
    replacements = {
        "bae-central-nll-mean": "Central BAE, " + r"$E_{\theta}$" + "(NLL)",
        "bae-coalition-nll-mean": "Coalition BAE, " + r"$E_{\theta}$" + "(NLL)",
        "bae-central-nll-var": "Central BAE, " + r"$Var_{\theta}$" + "(NLL)",
        "bae-coalition-nll-var": "Coalition BAE, " + r"$Var_{\theta}$" + "(NLL)",
        "ae-central-nll-mean": "Central AE, " + r"$E_{\theta}$" + "(NLL)",
        "ae-coalition-nll-mean": "Coalition AE, " + r"$E_{\theta}$" + "(NLL)",
        "ae-central-gradshap-mean": "Central AE, GradShap",
        "ae-central-deeplift-mean": "Central AE, DeepLift"
    }

    df_perf_new = df_perf.copy()
    for key,val in replacements.items():
        df_perf_new.replace(key, val, inplace=True)
    return df_perf_new

def replace_full_name_list(model_list):
    replacements = {
        "bae-central-nll-mean": "Central BAE, " + r"$E_{\theta}$" + "(NLL)",
        "bae-coalition-nll-mean": "Coalition BAE, " + r"$E_{\theta}$" + "(NLL)",
        "bae-central-nll-var": "Central BAE, " + r"$Var_{\theta}$" + "(NLL)",
        "bae-coalition-nll-var": "Coalition BAE, " + r"$Var_{\theta}$" + "(NLL)",
        "ae-central-nll-mean": "Central AE, " + r"$E_{\theta}$" + "(NLL)",
        "ae-coalition-nll-mean": "Coalition AE, " + r"$E_{\theta}$" + "(NLL)",
        "ae-central-gradshap-mean": "Central AE, GradShap",
        "ae-central-deeplift-mean": "Central AE, DeepLift"
    }

    return [replacements.get(x, x) for x in model_list]

def aggregate_perf_df(df_perf, perf_key = "pearson", apply_abs=True):
    df_perf_temp = df_perf.copy()
    df_perf_temp["explanation"] = df_perf_temp["explanation"] + "-" + df_perf_temp["mean-var"]

    df_perf_filtered = df_perf_temp[df_perf_temp["perf_name"] == perf_key]
    if apply_abs:
        df_perf_filtered.loc[:,"perf_score"] = df_perf_filtered.loc[:,"perf_score"].abs()

    for aggregate_mode in ["mean", "sem"]:
        for column in ["model_capacity", "n_layers","total_model_cap"]:
            print(aggregate_mode + "-" + column+"-"+perf_key)
            for i, model_cap in enumerate(df_perf_filtered[column].unique()):
                df_perf_filtered_ = df_perf_filtered[df_perf_filtered[column] == model_cap]


                if aggregate_mode == "mean":
                    aggregated_df_temp = (
                        df_perf_filtered_.groupby(["dataset", "bae_or_ae", "bae_config", "explanation"]).mean())
                else:
                    aggregated_df_temp = (
                        df_perf_filtered_.groupby(["dataset", "bae_or_ae", "bae_config", "explanation"]).sem())

                if i > 0:
                    aggregated_df[str(model_cap)] = aggregated_df_temp["perf_score"]
                else:
                    aggregated_df = aggregated_df_temp.copy()
                    aggregated_df[str(model_cap)] = aggregated_df_temp["perf_score"]
            aggregated_df = aggregated_df.drop(["perf_score"], axis=1)
            aggregated_df = aggregated_df.round(2)
            print(aggregated_df)

def get_perc_high_corr(df_perf, dataset="PRONOSTIA", high_corr_threshold = 0.85, total_model_cap=None, comparison="greater", inverse=False):
    perf_key = "pearson"
    df_perf_filtered = df_perf[df_perf["perf_name"] == perf_key].copy()
    results = {}
    for model_type in df_perf_filtered["model_type"].unique():
        # print(model_type)
        df_perf_filtered_ = df_perf_filtered[df_perf_filtered["model_type"] == model_type]
        df_perf_filtered_ = df_perf_filtered_[df_perf_filtered_["dataset"] == dataset]
        # df_perf_filtered_ = df_perf_filtered_[df_perf_filtered_["model_capacity"]==str(model_cap)]
        # print(df_perf_filtered_)

        if total_model_cap is not None:
            df_perf_filtered_ = df_perf_filtered_[df_perf_filtered_["total_model_cap"] == str(total_model_cap)]

        # print(df_perf_filtered_)
        # print(df_perf_filtered_.columns)
        total_len = len(df_perf_filtered_)
        if comparison == "greater":
            n_high_corr = len(df_perf_filtered_[df_perf_filtered_["perf_score"].abs() >= high_corr_threshold])
        else:
            n_high_corr = len(df_perf_filtered_[df_perf_filtered_["perf_score"].abs() <= high_corr_threshold])

        # perc_high_corr = n_high_corr / total_len * 100
        perc_high_corr = n_high_corr / total_len
        if inverse:
            perc_high_corr = 1-perc_high_corr
        results.update({model_type: perc_high_corr})
    res = pd.DataFrame([results]).round(2).T
    # res.columns = [dataset]
    return res

def get_perc_high_corrs(df_perf, datasets=["PRONOSTIA","ZEMA"],
                        threshold_ranges=[0.7,0.75,0.8,0.85,0.9,0.95,1.0],
                        comparison="greater"
                        , inverse=False
                        ):
    """
    Gets percentage of high correlation (with varying thresholds) on each dataset.

    Parameters
    ----------
    df_perf
        Main df results
    datasets
        List of dataset names
    threshold_ranges
        Threshold list to vary

    Returns
    -------
    Dictionary of percentages of high correlations keyed by each dataset

    """
    result_dict = {}
    for dataset in datasets:
        df_high_corrs = pd.concat([get_perc_high_corr(df_perf=df_perf,
                                                      dataset=dataset,
                                                      high_corr_threshold=float(threshold),
                                                      comparison=comparison,
                                                      inverse = inverse
                                                      )
                                   for threshold in threshold_ranges], axis=1)
        df_high_corrs.columns = threshold_ranges
        result_dict.update({dataset:df_high_corrs.copy()})

    return result_dict

def plot_high_perc_corr(df_high_corrs, threshold_ranges_label=[0.7,0.75,0.8,0.85,0.9,0.95,1.0],ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)
    ax_traces = []
    for row in list(df_high_corrs.index):
        if "coalition" in row:
            marker = "--"
        else:
            marker = "-"
        ax_traces += ax.plot(df_high_corrs.columns, df_high_corrs.loc[row], marker)
        ax.scatter(threshold_ranges_label, df_high_corrs.loc[row,threshold_ranges_label],
                   s=8)

    # plt.legend(df_high_corrs.index)
    ax.set_xticks(threshold_ranges_label)
    ax.set_xticklabels([str(round(float(label), 1)) if (float(label) != 0.0) else str(int(label)) for label in threshold_ranges_label])
    # ax.set_yticks(np.arange(0,110,10))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([str(round(float(label), 1)) if (float(label) != 0.0) else str(int(label)) for label in threshold_ranges_label])

    return ax_traces, list(df_high_corrs.index)

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
pd.set_option('display.max_rows', 100)
main_filename = "MLEXP-Explainability/unsupervised-BAE.p"
publication_folder = "publication/"
df_perf = load_bae_results(filename = main_filename)


# threshold_ranges = np.arange(0.7,1.0,0.01)
# threshold_ranges_label = np.arange(0.7,1.0,0.05)

# threshold_ranges = np.arange(0.,1.0+0.01,0.01)
threshold_ranges = ["%.2f" % i for i in np.arange(0.,1.0+0.01,0.01)]
threshold_ranges_label = ["%.2f" % i for i in np.arange(0.,1.0+0.1,0.1)]

threshold_ranges[0] = "0"
threshold_ranges_label[0] = "0"
threshold_ranges[-1] = "1"
threshold_ranges_label[-1] = "1"
# threshold_ranges_label = np.arange(0.01,1.0,0.05)

high_corr_dict = get_perc_high_corrs(df_perf, threshold_ranges=threshold_ranges, comparison="lower", inverse=True)

figsize = (10,3)
fig, (ax1,ax2) = plt.subplots(1,2, figsize=figsize)
plot_high_perc_corr(high_corr_dict["PRONOSTIA"], ax=ax1, threshold_ranges_label=threshold_ranges_label)
ax2_traces, model_names = plot_high_perc_corr(high_corr_dict["ZEMA"], ax=ax2, threshold_ranges_label=threshold_ranges_label)

# ax1.set_ylabel("p (|Pearson "+r"$(S^{drift} , S^{\bot{drift}})$"+ r"$\geq{} \rho{}$"+")")
# ax1.set_ylabel("p (|Pearson "+r"$(S^{drift} , S^{\bot{drift}}) \geq{} \rho{} )$")
ax1.set_ylabel(r"$\mathrm{p} \, (\:\| \mathrm{Pearson}\, (S^{drift} , S^{\bot{drift}})| \geq{r}\: )$")
ax1.set_ylabel(r"$\mathrm{p} \, (\:\| \mathrm{Pearson}\,| \geq{r}\: )$")

ax1.set_xlabel(r"$r$")
ax2.set_xlabel(r"$r$")
ax1.yaxis.grid()
ax2.yaxis.grid()
model_names = replace_full_name_list(model_names)
ax2.legend(ax2_traces,model_names,loc='center left', bbox_to_anchor=(1, 0.5))

# ax1.vlines(0.7*100,-10,105)
# ax1.vlines(0.7*100)
ax1.axvline(x=0.8*100, color="black")
ax2.axvline(x=0.8*100, color="black")
ax1.set_title("(a) PRONOSTIA")
ax2.set_title("(b) ZEMA")

fig.tight_layout()
fig.savefig(publication_folder+"pcorr.png",dpi=500)






