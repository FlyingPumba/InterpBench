import os
import pickle
import pandas as pd
from utils.bad_runs import bad_runs

def walk_dirs_and_get_scores(weight = 510, algorithm = "acdc"):
    if algorithm == "acdc":
        return get_acdc_scores(weight)
    elif "sp" in algorithm:
        return get_sp_scores(weight, algorithm)

def get_realism_scores(weight, algorithm):
    if weight is None:
        weight = ""
    weight = str(weight)
    if algorithm == "acdc":
        return get_acdc_realism_scores(weight)
    elif "sp" in algorithm:
        return get_sp_realism_scores(weight, algorithm)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")

def append_row(table, row):
    return pd.concat([
                table, 
                pd.DataFrame([row], columns=row.index)]
        ).reset_index(drop=True)

def get_acdc_scores(weight):
    # make an empty table with columns: run, threshold, tpr, fpr
    df = pd.DataFrame(columns=["run", "threshold", "node_tpr", "node_fpr", "edge_tpr", "edge_fpr"])

    
    for folder in os.listdir("results"):
        if "acdc" not in folder:
            continue
        weight_folder = os.path.join("results", folder, f"weight_{weight}")
        if not os.path.exists(weight_folder):
            continue
        print(weight_folder)
        for thresholds_folder in os.listdir(weight_folder):
            results_file = os.path.join(weight_folder, thresholds_folder, "result.pkl")
            if not os.path.exists(results_file):
                continue
            run = folder.split("_")[-1]
            if run in bad_runs:
                print(f"Skipping {run}")
                continue
            threshold = float(thresholds_folder.split("_")[-1])
            result = pickle.load(open(results_file, "rb"))
            entry = {"run": run, "threshold": threshold, 
                    "node_tpr": result["nodes"]["tpr"],
                    "node_fpr": result["nodes"]["fpr"],
                    "edge_tpr": result["edges"]["tpr"],
                    "edge_fpr": result["edges"]["fpr"]}
            if 'N/A' in entry.values():
                print(f"Skipping {run} {threshold}")
                continue
            entry = pd.Series(entry)
            df = append_row(df, entry)
    return df

def get_sp_scores(weight, algorithm):
    df = pd.DataFrame(columns=["run", "lambda", "node_tpr", "node_fpr", "edge_tpr", "edge_fpr"])
    for folder in os.listdir("results"):
        if algorithm not in folder:
            continue
        weight_folder = os.path.join("results", folder, f"weight_{weight}")
        if not os.path.exists(weight_folder):
            continue
        print(weight_folder)
        for lambda_folder in os.listdir(weight_folder):
            results_file = os.path.join(weight_folder, lambda_folder, "result.pkl")
            if not os.path.exists(results_file):
                continue
            run = folder.split("_")[-1]
            if run in bad_runs:
                print(f"Skipping {run}")
                continue
            threshold = float(lambda_folder.split("_")[-1])
            result = pickle.load(open(results_file, "rb"))
            entry = {"run": run, "lambda": threshold, 
                    "node_tpr": result["nodes"]["tpr"],
                    "node_fpr": result["nodes"]["fpr"],
                    "edge_tpr": result["edges"]["tpr"],
                    "edge_fpr": result["edges"]["fpr"]}
            if 'N/A' in entry.values():
                print(f"Skipping {run} {threshold}")
                continue
            entry = pd.Series(entry)
            df = append_row(df, entry)
    return df
    
def get_acdc_realism_scores(weight = ""):
    project = "node_realism"
    import wandb 
    api = wandb.Api()
    runs = api.runs(f"{project}")
    df = pd.DataFrame(columns = ["run", "threshold", "score", "weights"])
    for run in runs:
        if "acdc" in run.group:
            if weight not in run.group:
                continue
            case = run.group.split("_")[1]
            if case in bad_runs:
                continue
            weights = (run.group.split("_")[-1])
            threshold = float(run.name)
            score = run.summary["score"]
            entry = pd.Series({"run": case, "threshold": threshold, "score": score, "weights": weights})
            df = append_row(df, entry)
    return df

def get_sp_realism_scores(weight = "", algorithm = "node_sp"):
    project = "node_realism"
    import wandb 
    api = wandb.Api()
    runs = api.runs(f"{project}", filters={"state": "finished"})
    df = pd.DataFrame(columns = ["run", "lambda", "score", "weights"])
    for run in runs:
        if algorithm in run.group:
            if weight not in run.group:
                continue
            case = run.group.split("_")[2]
            if case in bad_runs:
                continue
            weights = (run.group.split("_")[-1])
            threshold = float(run.name)
            score = run.summary["score"]
            entry = pd.Series({"run": case, "lambda": threshold, "score": score, "weights": weights})
            df = append_row(df, entry)
    return df