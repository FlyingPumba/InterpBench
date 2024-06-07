import os
import json
import pickle

import pandas as pd
import huggingface_hub as hf

from circuits_benchmark.utils.get_cases import get_cases

hf_fs = hf.HfFileSystem()
hf_repo_id = "cybershiptrooper/InterpBench"


def build_metadata():
  metadata = load_benchmark_base_metadata()
  cases_info = []

  files_per_case = get_files_per_case_in_hf()
  case_ids = files_per_case.keys()
  print(f"Found {len(case_ids)} cases in Hugging Face")

  for case_id in case_ids:
    case_info = build_case_info(case_id, files_per_case)
    cases_info.append(case_info)

  metadata["cases"] = cases_info
  write_benchmark_metadata_json(metadata)
  write_cases_metadata(cases_info)

def write_cases_metadata(cases_info):
  # flatten transformer_cfg and training_args, and remove files and vocab
  for case_info in cases_info:
    if "training_args" in case_info:
      training_arg_keys = case_info["training_args"].keys()
      for key in training_arg_keys:
        case_info[f"training_args.{key}"] = case_info["training_args"][key]
      del case_info["training_args"]

    if "transformer_cfg" in case_info:
      cfg_keys = case_info["transformer_cfg"].keys()
      for key in cfg_keys:
        case_info[f"transformer_cfg.{key}"] = case_info["transformer_cfg"][key]
      del case_info["transformer_cfg"]

    if "files" in case_info:
      del case_info["files"]

    if "vocab" in case_info:
      del case_info["vocab"]

  # convert cases_info to pandas dataframe, and output to csv and parquet
  df = pd.DataFrame(cases_info)
  df.to_csv('benchmark_cases_metadata.csv', index=False)
  df.to_parquet('benchmark_cases_metadata.parquet', index=False)

def build_case_info(case_id, files_per_case):
  case_info = {
    "case_id": case_id,
    "files": files_per_case[case_id]
  }

  # Case description
  cases = get_cases(indices=[case_id])
  if len(cases) == 0:
    print(f"WARNING: No case found for case {case_id}")
  else:
    case = cases[0]
    case_info["task_description"] = case.get_task_description()
    case_info["vocab"] = list(case.get_vocab())
    case_info["max_seq_len"] = case.get_max_seq_len()
    case_info["min_seq_len"] = case.get_min_seq_len()

  # Model architecture info
  cfg_pkl = [f for f in files_per_case[case_id] if "_cfg_" in f and f.endswith(".pkl")]
  if len(cfg_pkl) == 0:
    print(f"WARNING: No cfg pkl file found for case {case_id}")
  else:
    cfg_pkl_file_name = cfg_pkl[0]
    with hf_fs.open(f"{hf_repo_id}/{case_id}/{cfg_pkl_file_name}", 'rb') as f:
      cfg_dict = pickle.load(f)

    cfg_dict["dtype"] = str(cfg_dict["dtype"])
    case_info["transformer_cfg"] = cfg_dict

  # Training info
  meta_json = [f for f in files_per_case[case_id] if f.startswith("meta_") and f.endswith(".json")]
  if len(meta_json) == 0:
    print(f"WARNING: No meta json file found for case {case_id}")
  else:
    meta_json_file_name = meta_json[0]
    training_args_str = hf_fs.read_text(f"{hf_repo_id}/{case_id}/{meta_json_file_name}")
    training_args = json.loads(training_args_str)

    del training_args["device"]
    del training_args["wandb_suffix"]

    case_info["training_args"] = training_args

  return case_info


def get_files_per_case_in_hf():
  files_per_case = {}
  for file_info in hf.list_files_info(hf_repo_id):
    path = file_info.path
    if "/" in path:
      case_id = path.split("/")[0]
      file_name = path.split("/")[-1]
      if case_id not in files_per_case:
        files_per_case[case_id] = []
      files_per_case[case_id].append(file_name)
  return files_per_case


def load_benchmark_base_metadata():
  with open(f"{os.getcwd()}/benchmark_base_metadata.json", 'r') as f:
    metadata = json.load(f)
  return metadata


def write_benchmark_metadata_json(metadata):
  with open('benchmark_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    build_metadata()