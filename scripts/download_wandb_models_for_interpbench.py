import re
import shutil

import wandb
import sys
import os

if __name__ == "__main__":
  if len(sys.argv) < 3:
    raise ValueError("Usage: python download_wandb_artifact.py <project_name> <accuracy_threshold> [case_id]")

  output_dir = "/home/ivan/src/InterpBench"
  if not os.path.exists(output_dir):
    raise ValueError(f"Output directory {output_dir} does not exist")

  # first argument is project name, second is accuracy threshold
  project_name = sys.argv[1]
  acc_threshold = float(sys.argv[2])
  overwrite_existing = True

  # check if we have optional filter for case_id
  case_id_filter = None
  if len(sys.argv) == 4:
    case_id_filter = sys.argv[3]
    print(f"Filtering for case_id {case_id_filter}")

  api = wandb.Api()
  runs = api.runs(project_name)
  for run in runs:
    if run.state != "finished":
      continue

    # extract case_id from run name
    if not run.name.startswith("case-"):
      raise ValueError(f"Run name {run.name} does not start with 'case-'")
    case_id = run.name.split("-")[1]

    # check if we need to filter by case_id
    if case_id_filter is not None and case_id != case_id_filter:
      continue

    # are test metrics good enough?
    if run.summary["val/accuracy"] < acc_threshold or run.summary["val/IIA"] < acc_threshold or run.summary["val/strict_accuracy"] < acc_threshold:
        print(f"Skipping run {run.name} because test metrics are not good enough for threshold {acc_threshold}")
        continue

    case_output_folder = os.path.join(output_dir, case_id)
    if os.path.exists(case_output_folder):
        if overwrite_existing:
            print(f"Overwriting existing folder {case_output_folder}")
            shutil.rmtree(case_output_folder, ignore_errors=True)
        else:
            print(f"Skipping existing folder {case_output_folder}")
            continue

    os.makedirs(case_output_folder)

    print(f"Downloading files for run {run.name}")
    files = run.files()
    for file in files:
      file_relative_path = file.name
      if file_relative_path.startswith("ll_models/"):
        file.download(replace=True, root=case_output_folder)

        # move to case output folder
        old_file_path = os.path.join(case_output_folder, file_relative_path)
        file_name = file_relative_path.split("/")[-1]
        new_file_path = os.path.join(case_output_folder, file_name)
        shutil.move(old_file_path, new_file_path)

        # rename files if needed
        if re.match(r"ll_model_\d+\.pth", file_name):
            os.rename(new_file_path, os.path.join(case_output_folder, "ll_model.pth"))
        elif re.match(r"ll_model_cfg_\d+\.pkl", file_name):
            os.rename(new_file_path, os.path.join(case_output_folder, "ll_model_cfg.pkl"))
        elif re.match(r"meta_\d+\.json", file_name):
            os.rename(new_file_path, os.path.join(case_output_folder, "meta.json"))

      shutil.rmtree(os.path.join(case_output_folder, "ll_models"), ignore_errors=True)
