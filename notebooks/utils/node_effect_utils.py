import numpy as np
import pandas as pd
from circuits_benchmark.utils.iit.correspondence import TracrCorrespondence
from circuits_benchmark.transformers.circuit_node import CircuitNode
import iit.utils.index as index


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

    get_status = lambda df, status: {
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


def make_combined_df(results, results_tracr):
    df_combined = pd.concat([df.assign(run=run) for run, df in results.items()])
    df_combined_tracr = pd.concat(
        [df.assign(run=run) for run, df in results_tracr.items()]
    )
    unique_runs = list(df_combined_tracr["run"].unique())
    print(f"Unique runs: {unique_runs}")
    df_combined_tracr.reset_index(drop=True, inplace=True)
    df_combined.reset_index(drop=True, inplace=True)
    return df_combined, df_combined_tracr


def create_name(node):
    if "mlp" in node.name:
        return node.name
    if node.index is not None and node.index != index.Ix[[None]]:
        return f"{node.name}, head {str(node.index).split(',')[-2]}"
    else:
        return f"{node.name}, head [:]"


def find_circuit_node(node: CircuitNode, hl_ll_corr: dict):
    for hl_node in hl_ll_corr.keys():
        if hl_node == node:
            return hl_ll_corr[hl_node]
    return set()


def find_circuit_node_by_name(node_name: str, hl_ll_corr: dict):
    if "mlp" in node_name:
        iit_nodes = find_circuit_node(CircuitNode(node_name), hl_ll_corr)
    elif "attn" in node_name:
        hook_name, head_num = node_name.split(", head ")
        head_num = int(head_num)
        hl_node = CircuitNode(hook_name, index=head_num)
        iit_nodes = find_circuit_node(hl_node, hl_ll_corr)
    else:
        raise ValueError(f"Node name {node_name} not recognized")
    return iit_nodes


def find_run_in_cases(run, cases):
    for case in cases:
        if case.get_index() == run:
            return case
    return None


def remove_nodes_with_zero_effect(df_iit, df_tracr, df_kl, df_tracr_kl, cases):
    zero_effect_mask = df_tracr["resample_ablate_effect"] == 0
    # iterate through rows where status = "in_circuit"
    for i, zero_effect in zero_effect_mask.items():
        run = df_tracr.loc[i, "run"]
        if zero_effect and df_tracr.loc[i, "status"] == "in_circuit":
            node_name = df_tracr.loc[i, "node"]
            # if iit_df doesn't have this run, skip
            case = find_run_in_cases(run, cases)
            if run not in df_iit["run"].unique() or case is None:
                # print(f"Run {run} not in IIT df or unique runs list, skipping")
                continue
            print(f"Run {run} has node {node_name} with zero effect and in circuit")

            tracr_output = case.get_tracr_output()
            hl_ll_corr = TracrCorrespondence.from_output(case, tracr_output)
            iit_nodes = find_circuit_node_by_name(node_name, hl_ll_corr)
            if iit_nodes is None:
                raise ValueError(f"Node {node_name} not found in IIT circuit!")
            for iit_node in iit_nodes:
                node_str = create_name(iit_node)
                print(f"Removing node {node_str} from IIT circuit in run {run}")
                # find the row in IIT df that corresponds to this node
                iit_node_row = df_iit[
                    (df_iit["run"] == run) & (df_iit["node"] == node_str)
                ]
                if iit_node_row.empty:
                    raise ValueError(f"Node {node_str} not found in IIT df")
                iit_node_row_index = iit_node_row.index[0]
                df_iit.loc[iit_node_row_index, "status"] = "not_in_circuit"
            df_tracr.loc[i, "status"] = "not_in_circuit"

            # remove the corresponding row from KL df
            def remove_from_circuit(df, run, node_name):
                kl_row = df[(df["run"] == run) & (df["node"] == node_name)]
                if kl_row.empty:
                    assert run not in df["run"].unique()
                    return
                print(f"Removing node {node_name} from KL circuit in run {run}")
                kl_row_index = kl_row.index[0]
                df.loc[kl_row_index, "status"] = "not_in_circuit"

            remove_from_circuit(df_kl, run, node_name)
            remove_from_circuit(df_tracr_kl, run, node_name)


def make_plot_lists(
    in_circuit_list,
    not_in_circuit_list,
    tracr_in_circuit_list,
    tracr_not_in_circuit_list,
    df_combined,
    all_cases,
    key,
    both_iit=False,
) -> tuple[tuple[list, list], tuple[list, list]]:
    def get_key(df, _key):
        try:
            return df[_key]
        except KeyError:
            return pd.Series()

    not_in_circuit_iit_effect = []
    not_in_circuit_tracr_effect = []
    if not both_iit:
        for df in not_in_circuit_list:
            values = get_key(df, key).values
            not_in_circuit_iit_effect.extend(values)
            not_in_circuit_tracr_effect.extend([0] * len(values))
    else:
        for df in tracr_not_in_circuit_list:
            values = list(get_key(df, key).values)
            nodes = list(df["node"].values)
            runs = list(df["run"].values)
            for i, node in enumerate(nodes):
                run = runs[i]
                # get the corresponding node in IIT by name
                node = df_combined.groupby("run").get_group(run).groupby("node").get_group(node)
                iit_value = node[key].values[0]
                tracr_value = values[i]
                if run == '3':
                    tracr_value = 0
                not_in_circuit_iit_effect.append(iit_value)
                not_in_circuit_tracr_effect.append(tracr_value)
                

    in_circuit_iit_effect = []
    in_circuit_tracr_effect = []
    for df in tracr_in_circuit_list:
        values = list(get_key(df, key).values)
        nodes = list(df["node"].values)
        runs = list(df["run"].values)
        for i, node in enumerate(nodes):
            run = runs[i]
            if both_iit:
                # get the corresponding node in IIT by name
                # print(values, key, in_circuit_list)
                node = df_combined.groupby("run").get_group(run).groupby("node").get_group(node)
                iit_value = node[key].values[0]
                tracr_value = values[i]
                in_circuit_iit_effect.append(iit_value)
                in_circuit_tracr_effect.append(tracr_value)
                continue
            case = find_run_in_cases(run, all_cases)
            if case is None:
                continue
            tracr_output = case.get_tracr_output()
            hl_ll_corr = TracrCorrespondence.from_output(case, tracr_output)
            iit_nodes = find_circuit_node_by_name(node, hl_ll_corr)
            if iit_nodes is None:
                raise ValueError(f"Node {node} not found in IIT circuit!")
            for iit_node in iit_nodes:
                node_str = create_name(iit_node)
                iit_node_row = df_combined[
                    (df_combined["run"] == run) & (df_combined["node"] == node_str)
                ]
                if iit_node_row.empty:
                    raise ValueError(f"Node {node_str} not found in IIT df")
                iit_node_row_index = iit_node_row.index[0]
                in_circuit_iit_effect.append(df_combined.loc[iit_node_row_index, key])
                in_circuit_tracr_effect.append(values[i])
    return (in_circuit_iit_effect, in_circuit_tracr_effect), (
        not_in_circuit_iit_effect,
        not_in_circuit_tracr_effect,
    )
