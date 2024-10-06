import json
import os
import pickle

import huggingface_hub as hf
import mlcroissant as mlc
import pandas as pd
from huggingface_hub import list_repo_files
from transformer_lens import HookedTransformerConfig

from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases

hf_fs = hf.HfFileSystem()
hf_repo_id = "cybershiptrooper/InterpBench"

PANDAS_TO_CROISSANT_VALUE_TYPE = {
    "string": mlc.DataType.TEXT,
    "int64": mlc.DataType.INTEGER,
    "float64": mlc.DataType.FLOAT,
    "bool": mlc.DataType.BOOL
}

LOCAL_REPO = "/home/ivan/src/InterpBench"
build_for_local_repo = True


def build_metadata():
  metadata = load_benchmark_base_metadata()
  cases_info = []

  if not build_for_local_repo:
    files_per_case = get_files_per_case_in_hf()
    case_ids = files_per_case.keys()
    print(f"Found {len(case_ids)} cases in Hugging Face")
  else:
    # Case ids are the directories in the local repo
    case_ids = [d for d in os.listdir(LOCAL_REPO) if os.path.isdir(os.path.join(LOCAL_REPO, d)) and d != ".git"]
    files_per_case = {case_id: os.listdir(f"{LOCAL_REPO}/{case_id}") for case_id in case_ids}
    print(f"Found {len(case_ids)} cases in local repo")

  for case_id in case_ids:
    case_info = build_case_info(case_id, files_per_case)
    cases_info.append(case_info)

  metadata["cases"] = cases_info
  write_benchmark_metadata_json(metadata)
  df_cases_info = write_cases_metadata(cases_info)
  write_benchmark_metadata_croissant(metadata, df_cases_info)


def flatten_dict_in_place(d):
  """Converts all list values to comma separated strings, and recursively flattens all nested dictionaries."""
  keys = list(d.keys())
  for k in keys:
    v = d[k]
    if isinstance(v, dict):
      flatten_dict_in_place(v)
      nested_keys = list(v.keys())
      for nested_k in nested_keys:
        nested_v = v[nested_k]
        d[f"{k}.{nested_k}"] = nested_v
      del d[k]
    elif isinstance(v, list):
      d[k] = ",".join(map(str, v))

def write_cases_metadata(cases_info):
  # flatten transformer_cfg and training_args, and remove files and vocab
  for case_info in cases_info:
    if "files" in case_info:
      del case_info["files"]

    if "vocab" in case_info:
      del case_info["vocab"]

    flatten_dict_in_place(case_info)

  # delete keys that have None value in all cases
  keys_to_delete = []
  for key in set([key for case_info in cases_info for key in case_info.keys()]):
    if all(key not in case_info or case_info.get(key) is None for case_info in cases_info):
      keys_to_delete.append(key)
  for key in keys_to_delete:
    for case_info in cases_info:
      if key in case_info:
        del case_info[key]

  # convert cases_info to pandas dataframe, and output to csv and parquet
  df = pd.DataFrame(cases_info)

  # fix object columns type
  for col in df.select_dtypes(include=['object']).columns:
    non_none_values = [case_info[col] for case_info in cases_info if col in case_info and case_info[col] is not None]
    if len(non_none_values) == 0:
      continue

    # assert they all have the same type
    assert all(isinstance(value, type(non_none_values[0])) for value in non_none_values)

    if isinstance(non_none_values[0], str):
      df[col] = df[col].astype("string")
    elif isinstance(non_none_values[0], bool):
      df[col] = df[col].astype(bool)

  df.to_csv('benchmark_cases_metadata.csv', index=False)
  df.to_parquet('benchmark_cases_metadata.parquet', index=False)

  return df


def build_case_info(case_id, files_per_case):
  case_info = {
    "case_id": case_id,
    "url": f"https://huggingface.co/{hf_repo_id}/tree/main/{case_id}",
  }

  # Case description and basic info
  cases = get_cases(indices=[case_id])
  assert len(cases) == 1, f"Case {case_id} not found or multiple cases found for the same name"

  case = cases[0]
  case_info["task_description"] = case.get_task_description()

  if isinstance(case, TracrBenchmarkCase):
    case_info["vocab"] = list(sorted(case.get_vocab()))
    case_info["max_seq_len"] = case.get_max_seq_len()
    case_info["min_seq_len"] = case.get_min_seq_len()
  elif "ioi" in case_id:
    case_info["max_seq_len"] = 16
    case_info["min_seq_len"] = 16
  else:
    raise ValueError(f"Unknown case type for case {case_id}")

  # Files
  case_info["files"] = []
  for file in sorted(files_per_case[case_id]):
    case_info["files"].append({
      "file_name": file,
      "url": f"https://huggingface.co/{hf_repo_id}/blob/main/{case_id}/{file}",
    })

  # Model architecture info
  cfg_pkl_file_name = "ll_model_cfg.pkl"
  if cfg_pkl_file_name not in files_per_case[case_id]:
    print(f"WARNING: No cfg pkl file found for case {case_id}")
  else:
    if not build_for_local_repo:
      with hf_fs.open(f"{hf_repo_id}/{case_id}/{cfg_pkl_file_name}", 'rb') as f:
        cfg = pickle.load(f)
    else:
      with open(f"{LOCAL_REPO}/{case_id}/{cfg_pkl_file_name}", 'rb') as f:
        cfg = pickle.load(f)

    if isinstance(cfg, dict):
      cfg_dict = cfg
    elif isinstance(cfg, HookedTransformerConfig):
      cfg_dict = cfg.to_dict()
    else:
      raise ValueError(f"Unknown type for cfg: {type(cfg)}")

    cfg_dict["dtype"] = str(cfg_dict["dtype"])
    if "original_architecture" in cfg_dict and cfg_dict["original_architecture"] is not None:
      cfg_dict["original_architecture"] = str(cfg_dict["original_architecture"])
    if "device" in cfg_dict:
      del cfg_dict["device"]

    case_info["transformer_cfg"] = cfg_dict
    case_info["transformer_cfg_file_url"] = f"https://huggingface.co/{hf_repo_id}/blob/main/{case_id}/{cfg_pkl_file_name}"

  # Training info
  meta_json_file_name = "meta.json"
  if meta_json_file_name not in files_per_case[case_id]:
    print(f"WARNING: No meta json file found for case {case_id}")
  else:
    if not build_for_local_repo:
      training_args_str = hf_fs.read_text(f"{hf_repo_id}/{case_id}/{meta_json_file_name}")
    else:
      with open(f"{LOCAL_REPO}/{case_id}/{meta_json_file_name}", 'r') as f:
        training_args_str = f.read()

    training_args = json.loads(training_args_str)

    if "device" in training_args:
      del training_args["device"]

    if "wandb_suffix" in training_args:
      del training_args["wandb_suffix"]

    if "wandb_name" in training_args:
      del training_args["wandb_name"]

    if "wandb_project" in training_args:
      del training_args["wandb_project"]

    case_info["training_args"] = training_args
    case_info["training_args_file_url"] = f"https://huggingface.co/{hf_repo_id}/blob/main/{case_id}/{meta_json_file_name}"

  weights_pkl_file_name = "ll_model.pth"
  if weights_pkl_file_name not in files_per_case[case_id]:
    print(f"WARNING: No weights pkl file found for case {case_id}")
  else:
    case_info["weights_file_url"] = f"https://huggingface.co/{hf_repo_id}/blob/main/{case_id}/{weights_pkl_file_name}"

  edges_pkl_file_name = "edges.pkl"
  if edges_pkl_file_name not in files_per_case[case_id]:
    print(f"WARNING: No edges pkl file found for case {case_id}")
  else:
    case_info["circuit_file_url"] = f"https://huggingface.co/{hf_repo_id}/blob/main/{case_id}/{edges_pkl_file_name}"

  return case_info


def get_files_per_case_in_hf():
  files_per_case = {}
  # Get the list of files in the repository
  file_paths = list_repo_files(hf_repo_id)

  for path in file_paths:
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


def write_benchmark_metadata_croissant(metadata, df_cases_info):
  distribution = [
    mlc.FileObject(
      id="hf-repository",
      name="hf-repository",
      description="The Hugging Face git repository.",
      content_url="https://huggingface.co/cybershiptrooper/InterpBench",
      encoding_format="git+https",
      sha256="main",
    ),
    mlc.FileObject(
      id="benchmark-cases-parquet",
      name="benchmark-cases-parquet",
      description="Parquet file describing all the cases in the benchmark.",
      contained_in=["hf-repository"],
      encoding_format="application/x-parquet",
    ),
    mlc.FileSet(
      id="training-args",
      name="training-args",
      description="Training arguments.",
      contained_in=["hf-repository"],
      encoding_format="application/json",
      includes="*/meta_[0-9]*.json",
    ),
    mlc.FileSet(
      id="circuits",
      name="circuits",
      description="Ground truth circuits (Pickle).",
      contained_in=["hf-repository"],
      encoding_format="application/octet-stream",
      includes="*/edges.pkl",
    ),
    mlc.FileSet(
      id="weights",
      name="weights",
      description="Serialized PyTorch state dictionaries (Pickle).",
      contained_in=["hf-repository"],
      encoding_format="application/octet-stream",
      includes="*/ll_model_[0-9]*.pkl",
    ),
    mlc.FileSet(
      id="cfgs",
      name="cfgs",
      description="Architecture configs (Pickle).",
      contained_in=["hf-repository"],
      encoding_format="application/octet-stream",
      includes="*/ll_model_cfg_[0-9]*.pkl",
    ),
  ]

  cases_fields = []
  for column, dtype in df_cases_info.dtypes.to_dict().items():
    cases_fields.append(
      mlc.Field(
        id=column,
        name=column,
        description=f"Column '{column}' from the parquet file describing all the cases in the benchmark.",
        data_types=PANDAS_TO_CROISSANT_VALUE_TYPE.get(str(dtype)),
        source=mlc.Source(
          file_set="benchmark-cases-parquet",
          extract=mlc.Extract(
            column=column,
          ),
        ),
      )
    )

  record_sets = [
    mlc.RecordSet(
      id="cases",
      name="cases",
      fields=cases_fields,
    )
  ]

  cr_metadata = mlc.Metadata(
    name=metadata["name"],
    description=metadata["description"],
    url=metadata["url"],
    license=metadata["license"],
    version=metadata["version"],
    distribution=distribution,
    record_sets=record_sets,
  )

  with open('benchmark_metadata_croissant.json', 'w') as f:
    json.dump(cr_metadata.to_json(), f, indent=2)

if __name__ == '__main__':
    build_metadata()