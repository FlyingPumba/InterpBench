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

    run_name = run.name # 'acdc-case-12-non-linear-compression-2-947e-07'
    case = run_name.split("-")[2]

    method = None
    if "non-linear-compression" in run_name:
      method = "non-linear"
    elif "linear-compression" in run_name:
      method = "linear"
    elif "natural-compression" in run_name:
      method = "natural"
    else:
      raise ValueError(f"Unknown method in run {run_name}")

    fpr = run.summary["edges_fpr"]
    tpr = run.summary["edges_tpr"]

    if case not in data_by_case:
      data_by_case[case] = {}

    if method not in data_by_case[case]:
      data_by_case[case][method] = {
        "fpr": [],
        "tpr": []
      }

    data_by_case[case][method]["fpr"].append(fpr)
    data_by_case[case][method]["tpr"].append(tpr)

  # Now we have the data, we can calculate the AUC and plot the ROC curve.
  for case, data_by_methods in data_by_case.items():
    # reset plot
    plt.clf()

    # create new figure and axis
    fig, ax = plt.subplots()

    for method, data in data_by_methods.items():
      # zip data together and sort it by fpr
      data = sorted(zip(data["fpr"], data["tpr"]))

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