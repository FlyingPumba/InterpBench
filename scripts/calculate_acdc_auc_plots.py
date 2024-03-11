from typing import Dict, List

import matplotlib.pyplot as plt
import wandb
from sklearn import metrics

if __name__ == "__main__":
  """Calculates the AUC for the ACDC dataset and plots the ROC curve."""
  api = wandb.Api()
  runs = api.runs("acdc-experiment")

  data_by_case: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
  for run in runs:
    if run.state != "finished":
      print(f"Skipping run {run.name} because it is not finished.")
      continue

    run_name = run.name
    case = run_name.split("case-")[1].split("-")[0]

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
      print(f"Skipping run {run.name} because it does not have the required metrics.")
      continue

    if "threshold" not in run.config:
      print(f"Skipping run {run.name} because it does not have the threshold in the config.")
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

      # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html
      auc = metrics.auc(fpr, tpr)
      print(f"Case {case} - Method {method}: AUC = {auc}")

      # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html
      display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
      display.plot(ax=ax, name=method)

    plt.savefig(f"acdc-case-{case}.png")
    plt.close(fig)