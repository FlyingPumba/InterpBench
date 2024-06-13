import pandas as pd
from utils.get_circuit_discovery_scores import append_row

def get_best_score(scores: pd.DataFrame, mean = False):
    best_scores = {}
    scores_by_run = scores.groupby("run")
    for run, run_scores in scores_by_run:
        best_score = run_scores['score'].max()
        if mean:
            best_score = run_scores['score'].mean()
        best_scores[run] = best_score
    return best_scores

def make_combined_realism_df(acdc_realism, node_sp_realism):
    combined_realism_df = pd.DataFrame()
    for k, v in acdc_realism.items():
        run = k
        acdc_realism_entry = v
        try:
            node_sp_realism_entry = node_sp_realism[k]
        except KeyError:
            print(f"Node sp realism not found for run {run}")
            continue
        entry = {"run": run, "acdc": acdc_realism_entry, "node_sp": node_sp_realism_entry}
        combined_realism_df = append_row(combined_realism_df, pd.Series(entry))
    return combined_realism_df

def make_combined_realism_df_from_list(list_of_realism_dfs, list_of_labels):
    # check if labels are unique
    assert len(list_of_labels) == len(set(list_of_labels)), "Labels are not unique"
    combined_realism_df = pd.DataFrame()
    for k, v in list_of_realism_dfs[0].items():
        run = k
        realism_entry = v
        realism_label = list_of_labels[0]
        entry = {"run": run, realism_label: realism_entry}
        for i in range(1, len(list_of_realism_dfs)):
            try:
                realism_entry = list_of_realism_dfs[i][k]
                realism_label = list_of_labels[i]
                entry[realism_label] = realism_entry
            except KeyError:
                print(f"Realism not found for run {run}")
                realism_entry = "N/A"
                realism_label = list_of_labels[i]
                entry[realism_label] = realism_entry
        combined_realism_df = append_row(combined_realism_df, pd.Series(entry))
    return combined_realism_df