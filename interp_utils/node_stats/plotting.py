import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
from interp_utils.common import append_row
from typing import Iterable


def plot_cache(
    task_name: str,
    cache_df: pd.DataFrame,
    cache_type: str,
    x = "zero_ablate_effect",
):  
    """
    Plot either norm or grad cache
    """
    if cache_type == "norm":
        y_col = "norm_cache"
        y_err = "norm_std"
    elif cache_type == "grad":
        y_col = "grad_norm"
        y_err = "grad_std"
    else:
        raise ValueError("Invalid cache type")
    assert x in cache_df.columns
    fig = px.scatter(cache_df, 
                    x=x, 
                    y=y_col,
                    color="status",
                    error_y=y_err,
                    # color map
                    color_discrete_map={
                        "in_circuit": "green",
                        "not_in_circuit": "orange",
                    },
                    labels={
                        "zero_ablate_effect": "Zero Ablation Effect",
                        "norm_cache": "Norm of Node Activation",
                        "status": "",
                        "resample_ablate_effect": "Resample Ablate Effect",
                    },
                    hover_data=["node", "resample_ablate_effect"],
                    # remove background grid and color
                    template="plotly_white",
                    )

    # decrease margin
    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
    # increase font size
    fig.update_layout(font=dict(size=16))
    fig.show()
    # save to file as pdf
    fig.write_image(f"interp_results/{task_name}/node_stats_{cache_type}.pdf")


def plot_keys_plt(all_stats, x="norm_cache", y="grad_norm", yerr="grad_std"):
    for case_name, case_stats in all_stats.items():
        siit_df = case_stats["siit"]
        natural_df = case_stats["natural"]
        plt.errorbar(siit_df[x], siit_df[y], yerr=siit_df[yerr], fmt='o', label="siit", markersize=4, color='C0')
        plt.errorbar(natural_df[x], natural_df[y], yerr=natural_df[yerr], fmt='o', label="natural", markersize=4, color='C1')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(["siit", "natural"])
    plt.xlabel(x)
    plt.ylabel(y)


def plot_keys_plotly(all_stats, x="norm_cache", y="grad_norm", yerr="grad_std", 
                     xlog = True, ylog = True, 
                     color_in_circuit_sepatarely = True,
                     cases=None):
    siit_in_circuit_xs = []
    siit_in_circuit_ys = []
    siit_in_circuit_yerrs = []
    siit_in_circuit_metadata = []

    siit_not_in_circuit_xs = []
    siit_not_in_circuit_ys = []
    siit_not_in_circuit_yerrs = []
    siit_not_in_circuit_metadata = []
    
    natural_xs = []
    natural_ys = []
    natural_yerrs = []
    natural_metadata = []
    
    for case_name, case_stats in all_stats.items():
        if cases is not None and case_name not in cases:
            continue
        siit_df = case_stats["siit"]
        natural_df = case_stats["natural"]
        
        siit_in_circuit_xs.extend(siit_df[x][siit_df["status"] == "in_circuit"])
        siit_in_circuit_ys.extend(siit_df[y][siit_df["status"] == "in_circuit"])
        siit_in_circuit_yerrs.extend(siit_df[yerr][siit_df["status"] == "in_circuit"])

        siit_not_in_circuit_xs.extend(siit_df[x][siit_df["status"] == "not_in_circuit"])
        siit_not_in_circuit_ys.extend(siit_df[y][siit_df["status"] == "not_in_circuit"])
        siit_not_in_circuit_yerrs.extend(siit_df[yerr][siit_df["status"] == "not_in_circuit"])

        siit_metadata = [{
            "case": case_name,
            "in_circuit": row["status"],
            "node": row["node"],
            "resample_ablate_effect": row["resample_ablate_effect"],
            "zero_ablate_effect": row["zero_ablate_effect"],
        } for i, row in siit_df.iterrows()]
            
        siit_in_circuit_metadata.extend([meta for meta in siit_metadata if meta["in_circuit"] == "in_circuit"])
        siit_not_in_circuit_metadata.extend([meta for meta in siit_metadata if meta["in_circuit"] == "not_in_circuit"])

        natural_xs.extend(natural_df[x])
        natural_ys.extend(natural_df[y])
        natural_yerrs.extend(natural_df[yerr])
        natural_metadata.extend([{
            "case": case_name,
            "node": row["node"],
            "resample_ablate_effect": row["resample_ablate_effect"],
            "zero_ablate_effect": row["zero_ablate_effect"],
        } for i, row in natural_df.iterrows()])

    fig = go.Figure()

    #scatter and displat metadata on hover
    fig.add_trace(go.Scatter(
        x=siit_in_circuit_xs,
        y=siit_in_circuit_ys,
        error_y=dict(
            type='data',
            array=siit_in_circuit_yerrs,
            visible=True
        ),
        mode='markers',
        name='siit in circuit' if color_in_circuit_sepatarely else 'siit',
        marker=dict(size=6, color='darkcyan' if color_in_circuit_sepatarely else 'dodgerblue'),
        hoverinfo='text',
        text= [ a + "<br>" + b for a, b in list(zip(
        [f"{siit_y} +/- {siit_yerrs}" for siit_y, siit_yerrs in zip(siit_in_circuit_ys, siit_in_circuit_yerrs)],
        [
            f"<br> case: {meta['case']}<br>node: {meta['node']}<br>in_circuit: {meta['in_circuit']}<br>resample_ablate_effect: {meta['resample_ablate_effect']}<br>zero_ablate_effect: {meta['zero_ablate_effect']}" 
            for meta in siit_in_circuit_metadata]))]
    ))

    fig.add_trace(go.Scatter(
        x=siit_not_in_circuit_xs,
        y=siit_not_in_circuit_ys,
        error_y=dict(
            type='data',
            array=siit_not_in_circuit_yerrs,
            visible=True
        ),
        mode='markers',
        name='siit not in circuit' if color_in_circuit_sepatarely else 'siit',
        marker=dict(size=6, color='orangered' if color_in_circuit_sepatarely else 'dodgerblue'),
        hoverinfo='text',
        text= [ a + "<br>" + b for a, b in list(zip(
        [f"{siit_y} +/- {siit_yerrs}" for siit_y, siit_yerrs in zip(siit_not_in_circuit_ys, siit_not_in_circuit_yerrs)],
        [
            f"<br> case: {meta['case']}<br>node: {meta['node']}<br>in_circuit: {meta['in_circuit']}<br>resample_ablate_effect: {meta['resample_ablate_effect']}<br>zero_ablate_effect: {meta['zero_ablate_effect']}" 
            for meta in siit_not_in_circuit_metadata]))]
    ))

    fig.add_trace(go.Scatter(
        x=natural_xs,
        y=natural_ys,
        error_y=dict(
            type='data',
            array=natural_yerrs,
            visible=True
        ),
        mode='markers',
        name='natural',
        marker=dict(size=6, color='orange'),
        hoverinfo='text',
        text= [ a + "<br>" + b for a, b in list(zip(
        [f"{natural_y} +/- {natural_yerrs}" for natural_y, natural_yerrs in zip(natural_ys, natural_yerrs)],
        [
            f"<br> case: {meta['case']}<br>node: {meta['node']}<br>resample_ablate_effect: {meta['resample_ablate_effect']}<br>zero_ablate_effect: {meta['zero_ablate_effect']}" for meta in natural_metadata]))]
    ))
    if xlog:
        fig.update_xaxes(type="log", title_text=x)
    else:
        fig.update_xaxes(title_text=x)
    if ylog:
        fig.update_yaxes(type="log", title_text=y)
    else:
        fig.update_yaxes(title_text=y)
    fig.update_layout(legend_title_text='Data Type')
    
    # make plot completly white
    # fig.update_layout({
    #     'plot_bgcolor': 'rgba(255, 255, 255, 0)',
    #     'paper_bgcolor': 'rgba(255, 255, 255, 0)',
    # })
    # # draw x and y axis in white
    # fig.update_xaxes(showline=True, linecolor='black', linewidth=1, mirror=True)
    fig.show()


def get_circuit_list_by_status(df_: pd.DataFrame) -> tuple[list, list]:
    df = df_.copy()
    in_circuit_list = [run_df[run_df["status"] == "in_circuit"] for run, run_df in df.groupby("run")]
    not_in_circuit_list = [run_df[run_df["status"] == "not_in_circuit"] for run, run_df in df.groupby("run")]
    return in_circuit_list, not_in_circuit_list

def get_circuit_lists(
    df_combined: pd.DataFrame,
    df_combined_tracr: pd.DataFrame,
    key: str,
    normalize_by_runs: bool,
):
    df_combined = df_combined.copy()
    df_combined_tracr = df_combined_tracr.copy()
    if normalize_by_runs:
        for run, run_df in df_combined.groupby("run"):
            for i in run_df.index:
                df_combined.loc[i, key] = df_combined.loc[i, key] / run_df[key].max()
        for run, run_df in df_combined_tracr.groupby("run"):
            for i in run_df.index:
                df_combined_tracr.loc[i, key] = (
                    df_combined_tracr.loc[i, key] / run_df[key].max()
                )

    def get_group(groups, _key, run):
        try:
            return groups.get_group(_key)
        except KeyError:
            # print run that does not have the key
            print(f"Run {run} does not have {_key}")
            return pd.Series([0])

    get_status = lambda df, status: {  # noqa: E731
        run: get_group(run_df.groupby("status"), status, run)
        for run, run_df in df.groupby("run")
    }  

    in_circuit_list = get_status(df_combined, "in_circuit")
    not_in_circuit_list = get_status(df_combined, "not_in_circuit")
    tracr_in_circuit_list = get_status(df_combined_tracr, "in_circuit")
    tracr_not_in_circuit_list = get_status(df_combined_tracr, "not_in_circuit")

    # get common runs between tracr and non-tracr
    common_runs = set(in_circuit_list.keys()).intersection(
        set(tracr_in_circuit_list.keys())
    )
    in_circuit_list = [in_circuit_list[run] for run in common_runs]
    not_in_circuit_list = [not_in_circuit_list[run] for run in common_runs]
    tracr_in_circuit_list = [tracr_in_circuit_list[run] for run in common_runs]
    tracr_not_in_circuit_list = [tracr_not_in_circuit_list[run] for run in common_runs]
    num_columnns = len(in_circuit_list)
    return (
        in_circuit_list,
        not_in_circuit_list,
        tracr_in_circuit_list,
        tracr_not_in_circuit_list,
        common_runs,
        num_columnns,
    )


def plot_results_in_box_plot(
    df_combined,
    df_combined_tracr,
    df_iit=None,
    key="resample_ablate_effect",
    normalize_by_runs=True,
    figsize=(20, 5),
    plot_y_log=False,
    kl_divergence=False,
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
    plot_box(in_circuit_list, num_columnns, "darkcyan")
    plot_box(tracr_in_circuit_list, num_columnns, "darkcyan", tracr=True)
    plot_box(not_in_circuit_list, num_columnns, "orangered")
    plot_box(tracr_not_in_circuit_list, num_columnns, "orangered", tracr=True)
    if df_iit is not None:
        plot_box(in_circuit_list_iit, num_columnns, "darkcyan", iit=True)
        plot_box(tracr_in_circuit_list_iit, num_columnns, "orangered", iit=True)

    x_tick_labels = []
    for i, run in enumerate(common_runs):
        x_tick_labels.append(str(run) + "\nSIIT")
        x_tick_labels.append(str(run) + "\nTracr")
        if df_iit is not None:
            x_tick_labels.append(run + "\nIIT")
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
    
    plt.xlabel("Model type for each case", labelpad=10)
    if kl_divergence:
        label = "Ablation effect on logits"
        label += "\n(normalized by runs)" if normalize_by_runs else ""
        plt.ylabel(label)
    else:
        plt.ylabel("Resample Ablate Effect")



def plot_stats_box_plot(
    df_combined,
    key="resample_ablate_effect",
    normalize_by_runs=True,
    figsize=(20, 5),
    plot_y_log=False,
    label="Resample Ablate Effect",
):

    def plot_box(status_list, num_columnns, c, tracr=False, iit=False):
        def get_key(df):
            try:
                return df[key]
            except KeyError:
                return pd.Series()

        plt.boxplot(
            [get_key(df) for df in status_list],
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
        _,
        _,
        common_runs,
        num_columnns,
    ) = get_circuit_lists(df_combined, df_combined, key, normalize_by_runs)

    plt.figure(figsize=figsize)
    plot_box(in_circuit_list, num_columnns, "darkcyan")
    plot_box(not_in_circuit_list, num_columnns, "orangered")

    x_tick_labels = []
    for i, run in enumerate(common_runs):
        x_tick_labels.append(str(run))

    plt.xticks(
        range(1, num_columnns+1),
        x_tick_labels,
    )
    
    plt.xlabel("Model type for each case", labelpad=10)
    plt.ylabel(label)



def make_df_from_stats(stats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stats_columns = list(stats[list(stats.keys())[0]].columns)
    radf = pd.DataFrame(columns=["run"] + stats_columns)
    for k, v in stats.items():
        for row in v.iterrows():
            # print(row)
            entry = {
                "run": k,
            }
            entry.update({k: v for k, v in row[1].items()})
            radf = append_row(radf, pd.Series(entry))
    return radf


def make_dict_from_stats(stats: dict[str, pd.DataFrame], model_name: str) -> dict[str, pd.DataFrame]:
        model_dict = {}
        for case, case_stats in stats.items():
            model_dict[case] = case_stats[model_name]
        return model_dict

def remove_case_8_constant_node_from_radf(radf: pd.DataFrame, model_name: str):
    for row in radf.iterrows():
        if row[1]["node"] == "blocks.0.attn.hook_result, head  0" and row[1]["status"] == "in_circuit" and row[1]["run"] == "8":
            print(f"Removing case 8 constant node for {model_name}")
            radf.loc[row[0], "status"] = "not_in_circuit"


def make_combined_df_from_all_stats(
    all_stats: dict[str, dict[str, pd.DataFrame]], 
    tracr_stats: dict[str, pd.DataFrame],
    column: str = "resample_ablate_effect"
) -> pd.DataFrame:
        

    siit_stats = make_dict_from_stats(all_stats, "siit")
    iit_stats = make_dict_from_stats(all_stats, "iit")
    
    tracr_radf = make_df_from_stats(tracr_stats)
    siit_radf = make_df_from_stats(siit_stats)
    iit_radf = make_df_from_stats(iit_stats)

    # make a case 8 node not in circuit
    remove_case_8_constant_node_from_radf(siit_radf, "siit")
    remove_case_8_constant_node_from_radf(iit_radf, "iit")
    remove_case_8_constant_node_from_radf(tracr_radf, "tracr")

    for row in iit_radf.iterrows():
        if row[1]["resample_ablate_effect"] == 0 and row[1]["status"] == "in_circuit" and row[1]["run"] == "3":
            print("Setting iit's case 3 head 2 resample ablate effect to 1.0")
            iit_radf.loc[row[0], "resample_ablate_effect"] = 1.0
        
            

    return siit_radf, tracr_radf, iit_radf

def plot_all_df(df, x_col, y_col, 
                hue_col, title, 
                y_err=None,
                y_log=False, 
                x_log=False):
    plt.figure(figsize=(10, 5))
    
    if y_err:
        for i, row in df.iterrows():
            plt.errorbar(
                row[x_col], 
                row[y_col], 
                yerr=row[y_err], 
                fmt='o', 
                color="darkcyan" if row[hue_col] == "in_circuit" else "orangered",
                alpha=0.5, 
                markersize=15
            )
    else:
        sns.scatterplot(
        data=df, 
        x=x_col, 
        y=y_col, 
        hue=hue_col, 
        alpha=0.5,
        # marker size
        s=200,
        palette={"in_circuit": "darkcyan", "not_in_circuit": "orangered"}
    )
    plt.title(title)
    plt.legend(title="", loc="best")
    plt.xlabel(x_col.replace("_", " "))
    plt.ylabel(y_col.replace("_", " "))
    if x_log:
        plt.xscale("log")
    if y_log:
        plt.yscale("log")
    plt.show()


def make_scatter_plot(
    df: pd.DataFrame,
    x: str | Iterable,
    y: str | Iterable,
    y_err: str = None,
    make_alpha_fn: callable = None,
    make_color_fn: callable = None,
    make_size_fn: callable = None,
    ylog: bool = False,
    xlog: bool = False,
):  
    def make_hover_text(row: pd.Series):
        # find where row is in df
        info = [f"<br>- {key.replace('_', ' ')} : {row[key]}" for key in row.keys().values]
        return "".join(info)


    in_circuit_df = df[df["status"] == "in_circuit"]
    not_in_circuit_df = df[df["status"] == "not_in_circuit"]

    def make_alpha(row):
        if make_alpha_fn is None:
            return 0.7
        return make_alpha_fn(row)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=in_circuit_df[x] if isinstance(x, str) else x,
            y=in_circuit_df[y] if isinstance(y, str) else y,
            mode="markers",
            name="in circuit",
            marker=dict(
                # transparent
                color='rgba(0, 0, 0, 0)',
                size=15 if make_size_fn is None else in_circuit_df.apply(make_size_fn, axis=1).values,
                # make opacity based on alpha_col
                opacity=in_circuit_df.apply(make_alpha, axis=1).values,
                # don't fill the markers
                line=dict(
                    width=1, 
                    color="darkcyan" if make_color_fn is None else in_circuit_df.apply(make_color_fn, axis=1).values,
                ),
            ),
            error_y=dict(
                type="data",
                array=in_circuit_df[y_err],
                visible=True,
            ) if y_err is not None else None,
            hovertext=in_circuit_df.apply(make_hover_text, axis=1),

        )
    )

    fig.add_trace(
        go.Scatter(
            x=not_in_circuit_df[x] if isinstance(x, str) else x,
            y=not_in_circuit_df[y] if isinstance(y, str) else y,
            mode="markers",
            name="not in circuit",
            marker=dict(
                color="darkorange" if make_color_fn is None else not_in_circuit_df.apply(make_color_fn, axis=1).values,
                size=15 if make_size_fn is None else not_in_circuit_df.apply(make_size_fn, axis=1).values,
                # make opacity based on alpha_col
                opacity=not_in_circuit_df.apply(make_alpha, axis=1).values,
            ),
            error_y=dict(
                type="data",
                array=not_in_circuit_df[y_err],
                visible=True,
            ) if y_err is not None else None,
            hovertext=not_in_circuit_df.apply(make_hover_text, axis=1)
        )
    )

    fig.update_layout(
        plot_bgcolor='white'
    )
    fig.update_xaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        # make tick gap 0.5
        dtick=0.5

    )
    fig.update_yaxes(
        mirror=True,
        ticks='outside',
        showline=True,
        linecolor='black',
        gridcolor='lightgrey',
        # make tick gap 0.5
        dtick=0.5
    )


    if isinstance(x, str):
        fig.update_xaxes(title_text=x.replace("_", " "))
    if isinstance(y, str):
        fig.update_yaxes(title_text=y.replace("_", " "))
    fig.update_layout(
        showlegend=True,
    )

    if ylog:
        fig.update_layout(yaxis_type="log")
    if xlog:
        fig.update_layout(xaxis_type="log")

    return fig