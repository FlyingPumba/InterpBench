import matplotlib.pyplot as plt
import numpy as np
from .node_effect_utils import get_circuit_lists
import pandas as pd
from .node_effect_utils import make_plot_lists

# make all font sizes bigger
plt.rcParams.update({"font.size": 13})

def pessimistic_auc(xs, ys):
    # Sort indices based on 'x' and 'y'
    i = np.lexsort(
        (ys, xs)
    )  # lexsort sorts by the last column first, then the second last, etc., i.e we firstly sort by x and then y to break ties

    xs = np.array(xs, dtype=np.float64)[i]
    ys = np.array(ys, dtype=np.float64)[i]

    # remove x and y values where y is not increasing
    while True:
        indices_to_remove = []
        for i in range(1, len(ys)):
            if ys[i] < ys[i - 1]:
                indices_to_remove.append(i)
        if len(indices_to_remove) == 0:
            break
        for i in indices_to_remove[::-1]:
            xs = np.delete(xs, i)
            ys = np.delete(ys, i)
    # prepend 0 and append 1
    xs = np.concatenate([[0], xs], [1])
    ys = np.concatenate([[0], ys], [1])

    dys = np.diff(ys)
    assert np.all(np.diff(xs) >= 0), "not sorted"
    assert np.all(dys >= 0), "not monotonically increasing"

    # The slabs of the stairs
    area = np.sum((1 - xs)[1:] * dys)
    return area


def plot_roc_curve(tprs, fprs, title, labels=None):
    plt.figure()
    lw = 2
    for i, rates in enumerate(zip(tprs, fprs)):
        i += 1
        tpr_list, fpr_list = rates
        auc = pessimistic_auc(fpr_list, tpr_list)
        # plot with colorscheme viridis
        cmap = plt.get_cmap("OrRd")
        label = labels[i - 1] if labels is not None else f"ROC curve {i}"
        plt.plot(
            fpr_list,
            tpr_list,
            lw=lw,
            label=f"{label} (area = {auc:.2f})",
            color=cmap(i * 80),
            marker="o",
        )
    # auc = metrics.auc(fprs, tprs)

    # plt.plot(fprs, tprs, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.3f)' % auc)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} ROC curve")
    plt.legend(loc="lower right")
    # increase font size of legend
    plt.legend(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # increase font size of axis labels
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.show()


def get_roc_curves_from_df(df, title, nodes=True, algorithm="acdc"):
    if algorithm == "acdc":
        sort_key = "threshold"
        ascending = False
    elif "sp" in algorithm:
        sort_key = "lambda"
        ascending = True
    tprs = []
    fprs = []
    labels = []
    for run, entries in df.groupby("run"):
        entries = entries.sort_values(sort_key, ascending=ascending)
        tpr = (
            entries["node_tpr"].tolist().copy()
            if nodes
            else entries["edge_tpr"].tolist().copy()
        )
        fpr = (
            entries["node_fpr"].tolist().copy()
            if nodes
            else entries["edge_fpr"].tolist().copy()
        )
        if len(tpr) < 2:
            continue
        assert len(tpr) == len(fpr)
        tprs.append(tpr)
        fprs.append(fpr)
        labels.append(run)
    plot_roc_curve(tprs, fprs, title, labels)


def plot_results_in_box_plot(
    df_combined,
    df_combined_tracr,
    df_iit=None,
    key="resample_ablate_effect",
    normalize_by_runs=True,
    figsize=(20, 5),
    plot_y_log=False,
    title=None,
):

    def plot_box(status_list, num_columnns, c, tracr=False, iit=False):
        def get_key(df):
            try:
                return df[key]
            except KeyError:
                return pd.Series()

        pos = 1 if tracr else 2 if iit else 0
        if iit:
            assert df_iit is not None, "df_iit is None, but iit is True"

        positions = (
            range(pos, num_columnns * 2, 2)
            if df_iit is None
            else range(pos, num_columnns * 3, 3)
        )
        plt.boxplot(
            [get_key(df) for df in status_list],
            positions=positions,
            patch_artist=True,
            showfliers=True,
            whis=[5, 95],
            boxprops=dict(facecolor=c, color=c),
            capprops=dict(color=c),
            whiskerprops=dict(color=c),
            flierprops=dict(color=c, markeredgecolor=c),
            medianprops=dict(color=c),
        )
        # plot y in log scale
        if plot_y_log:
            plt.yscale("log")

    (
        in_circuit_list,
        not_in_circuit_list,
        tracr_in_circuit_list,
        tracr_not_in_circuit_list,
        common_runs,
        num_columnns,
    ) = get_circuit_lists(df_combined, df_combined_tracr, key, normalize_by_runs)

    if df_iit is not None:
        (
            in_circuit_list_iit,
            not_in_circuit_list_iit,
            tracr_in_circuit_list_iit,
            tracr_not_in_circuit_list_iit,
            common_runs_iit,
            num_columnns_iit,
        ) = get_circuit_lists(df_iit, df_combined_tracr, key, normalize_by_runs)
        assert (
            common_runs == common_runs_iit
        ), f"Common runs are not the same {common_runs} != {common_runs_iit}"
        assert (
            num_columnns == num_columnns_iit
        ), f"Number of columns are not the same {num_columnns} != {num_columnns_iit}"

    plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title)
    else:
        plt.title(f"{key} for different runs")
    plot_box(in_circuit_list, num_columnns, "darkcyan")
    plot_box(tracr_in_circuit_list, num_columnns, "darkcyan", tracr=True)
    plot_box(not_in_circuit_list, num_columnns, "orangered")
    plot_box(tracr_not_in_circuit_list, num_columnns, "orangered", tracr=True)
    if df_iit is not None:
        plot_box(in_circuit_list_iit, num_columnns, "darkcyan", iit=True)
        plot_box(tracr_in_circuit_list_iit, num_columnns, "orangered", iit=True)

    x_tick_labels = []
    for i, run in enumerate(common_runs):
        x_tick_labels.append(run + "_siit")
        x_tick_labels.append(run + "_tracr")
        if df_iit is not None:
            x_tick_labels.append(run + "_iit")
    if df_iit is not None:
        plt.xticks(
            range(0, num_columnns * 3),
            x_tick_labels,
        )
    else:
        plt.xticks(
            range(0, num_columnns * 2),
            x_tick_labels,
        )

def plot_results_in_scatter_plot(
    results,
    results_tracr,
    all_cases,
    key="resample_ablate_effect",
    normalize_by_runs=True,
    figsize=(10, 5),
    plot_minmax_lines=True,
    mean=False,
    title="Comparing ablate effect for different runs",
    xlabel="tracr ablate effect",
    ylabel="iit ablate effect",
    both_iit=False,
):
    (
        in_circuit_list,
        not_in_circuit_list,
        tracr_in_circuit_list,
        tracr_not_in_circuit_list,
        _,
        _,
    ) = get_circuit_lists(results, results_tracr, key, normalize_by_runs)

    def get_key(df):
        try:
            return df[key]
        except KeyError:
            return pd.Series()

    if mean:
        iit_in_circuit_effect_list = [get_key(df).mean() for df in in_circuit_list]
        iit_not_in_circuit_effect_list = [
            get_key(df).mean() for df in not_in_circuit_list
        ]
        tracr_in_circuit_effect_list = [
            get_key(df).mean() for df in tracr_in_circuit_list
        ]
        tracr_not_in_circuit_effect_list = [
            get_key(df).mean() for df in tracr_not_in_circuit_list
        ]
    else:
        (iit_in_circuit_effect_list, tracr_in_circuit_effect_list), (
            iit_not_in_circuit_effect_list,
            tracr_not_in_circuit_effect_list,
        ) = make_plot_lists(
            in_circuit_list,
            not_in_circuit_list,
            tracr_in_circuit_list,
            tracr_not_in_circuit_list,
            df_combined=results,
            all_cases=all_cases,
            key=key,
            both_iit=both_iit,
        )
    plt.figure(figsize=figsize)
    plt.title(title)
    import numpy as np

    print(
        tracr_in_circuit_effect_list,
        iit_in_circuit_effect_list,
        tracr_not_in_circuit_effect_list,
        iit_not_in_circuit_effect_list,
    )
    plt.scatter(
        np.abs(tracr_in_circuit_effect_list),
        np.abs(iit_in_circuit_effect_list),
        color="darkcyan",
        label="iit_in_circuit",
        alpha=0.5,
        s=150,
    )
    print(len(tracr_not_in_circuit_effect_list), len(iit_not_in_circuit_effect_list))
    plt.scatter(
        np.abs(tracr_not_in_circuit_effect_list),
        np.abs(iit_not_in_circuit_effect_list),
        color="orangered",
        label="iit_not_in_circuit",
        alpha=0.2,
        s=150,
    )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", alpha=0.9)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legend_list = ["in_circuit", "not_in_circuit", "x=y line"]
    if plot_minmax_lines:
        plt.hlines(
            [min(iit_in_circuit_effect_list)],
            0,
            1,
            color="darkcyan",
            linestyle="--",
        )
        plt.hlines(
            [max(iit_not_in_circuit_effect_list)],
            0,
            1,
            color="maroon",
            linestyle="--",
        )
        # plot the gap
        gap_color = (
            "green"
            if min(iit_in_circuit_effect_list) > max(iit_not_in_circuit_effect_list)
            else "red"
        )
        plt.plot(
            [0.5, 0.5],
            [min(iit_in_circuit_effect_list), max(iit_not_in_circuit_effect_list)],
            color=gap_color,
            linestyle="--",
        )
        # fill the gap with color
        plt.fill_between(
            [0, 1],
            min(iit_in_circuit_effect_list),
            max(iit_not_in_circuit_effect_list),
            color=gap_color,
            alpha=0.2,
        )

        legend_list += ["min_in_circuit", "max_not_in_circuit", "gap"]

    plt.legend(legend_list)  # , loc='center left', bbox_to_anchor=(1, 0.7))
