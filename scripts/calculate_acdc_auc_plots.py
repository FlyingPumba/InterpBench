from typing import Dict, List

import matplotlib.pyplot as plt
import wandb
from sklearn import metrics

if __name__ == "__main__":
  """Calculates the AUC for the ACDC dataset and plots the ROC curve."""
  api = wandb.Api()
  runs = api.runs("acdc-experiment")

  skipped_runs_by_case: Dict[str, Dict[str, List[str]]] = {}
  for i in range(0, 40):
    skipped_runs_by_case[str(i)] = {
      "non-finished": [],
      "no-metrics": [],
      "no-threshold": []
    }

  runs_per_case: Dict[str, int] = {}
  for i in range(0, 40):
    runs_per_case[str(i)] = 0

  data_by_case: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
  for run in runs:
    run_name = run.name
    case = run_name.split("case-")[1].split("-")[0]
    runs_per_case[case] += 1

    if run.state != "finished":
      skipped_runs_by_case[case]["non-finished"].append(run.name)
      print(f"Skipping run {run.name} (Method: {method}, case: {case}) because it is not finished.")
      continue

    method = None
    if "non-linear-compression" in run_name:
      method = "non-linear"
    elif "linear-compression" in run_name:
      method = "linear"
    elif "natural-compression" in run_name:
      method = "natural"
    else:
      method = "tracr"

    if "edges_fpr" not in run.summary or "edges_tpr" not in run.summary:
      skipped_runs_by_case[case]["no-metrics"].append(run.name)
      print(f"Skipping run {run.name} (Method: {method}, case: {case}) because it does not have the required metrics.")
      continue

    if "threshold" not in run.config:
      skipped_runs_by_case[case]["no-threshold"].append(run.name)
      print(f"Skipping run {run.name} (Method: {method}, case: {case}) because it does not have the threshold in the config.")
      continue

    fpr = run.summary["edges_fpr"]
    tpr = run.summary["edges_tpr"]
    threshold = run.config["threshold"]

    print(f"Case {case} - Method {method} - Threshold {threshold}: FPR = {fpr}, TPR = {tpr}")

    if case not in data_by_case:
      data_by_case[case] = {}

    if method not in data_by_case[case]:
      data_by_case[case][method] = {
        "fpr": [],
        "tpr": [],
        "thresholds": []
      }

    if threshold in data_by_case[case][method]["thresholds"]:
      print(f"WARNING: Threshold {threshold} already exists for case {case} and method {method}.")

    data_by_case[case][method]["fpr"].append(fpr)
    data_by_case[case][method]["tpr"].append(tpr)
    data_by_case[case][method]["thresholds"].append(threshold)

  # Now we have the data, we can calculate the AUC and plot the ROC curve.
  cases = sorted(list(data_by_case.keys()))
  for case in cases:
    data_by_methods = data_by_case[case]

    # reset plot
    plt.clf()

    # create new figure and axis
    fig, ax = plt.subplots()

    methods = sorted(list(data_by_methods.keys())) # ['linear', 'natural', 'non-linear', 'tracr']
    for method in methods:
      data = data_by_methods[method]

      # zip data together and sort it by threshold
      data = sorted(zip(data["fpr"], data["tpr"], data["thresholds"]), key=lambda x: x[2], reverse=True)

      # split again into fpr and tpr
      fpr = [x[0] for x in data]
      tpr = [x[1] for x in data]

      # Fix non-decreasing values in fpr and tpr due to ACDC's non-deteministic behavior
      # I.e., the order in which edges are processed can change the results.
      total_fpr_fixed = 0
      for i in range(1, len(fpr)):
        if fpr[i] < fpr[i-1]:
          fpr[i] = fpr[i-1]
          total_fpr_fixed += 1
          print(f"Case {case} - Method {method}: Fixed fpr at index {i}")

      total_tpr_fixed = 0
      for i in range(1, len(tpr)):
        if tpr[i] < tpr[i-1]:
          tpr[i] = tpr[i-1]
          total_tpr_fixed += 1
          print(f"Case {case} - Method {method}: Fixed tpr at index {i}")

      print(f"Case {case} - Method {method}: Fixed a total of {total_fpr_fixed} fpr and {total_tpr_fixed} tpr values.")

      try:
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
        auc = metrics.auc(fpr, tpr)
        print(f"Case {case} - Method {method}: AUC = {auc}")

        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
        display.plot(ax=ax, name=method)
      except ValueError as e:
        print(f"Case {case} - Method {method}: AUC calculation failed: {e}")

    plt.savefig(f"acdc-case-{case}.png")
    plt.close(fig)

  print("Skipped runs by case:")
  for case, skipped_runs in skipped_runs_by_case.items():
    print(f"Case {case}:")
    print(f" - Total runs: {runs_per_case[case]}")
    for reason, runs in skipped_runs.items():
      print(f" - {reason}: {len(runs)} runs")