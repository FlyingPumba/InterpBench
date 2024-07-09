#!/usr/bin/env python3
import subprocess
import sys
from itertools import product
from math import ceil
from pathlib import Path
from typing import List

from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.iit.ll_cfg import compression_ratio_map

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
  JOB_TEMPLATE = f.read()

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands():
  # training_methods = ["linear-compression", "non-linear-compression", "natural-compression", "autoencoder"]
  training_methods = ["non-linear-compression"]
  case_instances = get_cases()

  cases = []
  compressed_d_model_size_by_case = {}
  compressed_d_head_size_by_case = {}
  for case in case_instances:
    case_name = case.get_name()

    if "ioi" in case_name:
      gt_model_cfg = case.get_ll_model().cfg
    else:
      gt_model_cfg = case.get_hl_model().cfg

    # Decide compressed d_model size
    if case_name in compression_ratio_map:
      compressed_d_model_size = ceil(gt_model_cfg.d_model / compression_ratio_map[case_name])
    else:
      compressed_d_model_size = ceil(gt_model_cfg.d_model / compression_ratio_map["default"])

    compressed_d_model_size = max(2, compressed_d_model_size)
    compressed_d_model_size_by_case[case_name] = compressed_d_model_size

    # Decide compressed d_head size
    compressed_d_head_size_by_case[case_name] = min(gt_model_cfg.d_head, max(1, compressed_d_model_size // gt_model_cfg.n_heads))

    cases.append(case_name)

  seeds = [67]
  lr_starts = [1e-2]

  linear_compression_args = {
    "linear-compression-initialization": ["linear"],  # ["orthogonal", "linear"],
  }

  autoencoder_args = {
    "ae-layers": [2],
    "ae-first-hidden-layer-shape": ["wide"],  # ["narrow", "wide"],
  }

  commands = []

  for method in training_methods:
    for case in cases:
        for seed in seeds:
          for lr_start in lr_starts:
            wandb_project = f"new-non-linear-compression"

            command = [
              ".venv/bin/python", "main.py",
              "train", method,
              f"-i={case}",
              f"--d-model={compressed_d_model_size_by_case[case]}",
              f"--d-head={compressed_d_head_size_by_case[case]}",
              f"--seed={seed}",
              f"--lr-start={lr_start}",
              "--early-stop-threshold=1",
              f"--wandb-project={wandb_project}",
            ]

            if method == "linear-compression":
              # produce all combinations of args in linear_compression_args
              arg_names = list(linear_compression_args.keys())
              arg_values = list(linear_compression_args.values())
              for arg_values_combination in product(*arg_values):
                specific_cmd = command.copy()
                for i, arg_name in enumerate(arg_names):
                  arg_value = arg_values_combination[i]
                  if arg_value == True:
                    specific_cmd.append(f"--{arg_name}") # just set the flag to trigger the store_true action
                  elif arg_value == False:
                    continue  # skip the argument so that we don't trigger the store_true action
                  else:
                    specific_cmd.append(f"--{arg_name}={arg_value}")

                if all("--wandb-name=" not in part for part in specific_cmd):
                  specific_cmd.append(f"--wandb-name={build_wandb_name(specific_cmd)}")

                commands.append(specific_cmd)

            if method == "non-linear-compression":
              specific_cmd = command.copy()
              if all("--wandb-name=" not in part for part in specific_cmd):
                specific_cmd.append(f"--wandb-name={build_wandb_name(specific_cmd)}")

              commands.append(specific_cmd)

            if method == "autoencoder":
              # produce all combinations of args in autoencoder_args
              arg_names = list(autoencoder_args.keys())
              arg_values = list(autoencoder_args.values())
              for arg_values_combination in product(*arg_values):
                specific_cmd = command.copy()
                for i, arg_name in enumerate(arg_names):
                  arg_value = arg_values_combination[i]
                  if arg_value == True:
                    specific_cmd.append(f"--{arg_name}") # just set the flag to trigger the store_true action
                  elif arg_value == False:
                    continue  # skip the argument so that we don't trigger the store_true action
                  else:
                    specific_cmd.append(f"--{arg_name}={arg_value}")

                if all("--wandb-name=" not in part for part in specific_cmd):
                  specific_cmd.append(f"--wandb-name={build_wandb_name(specific_cmd)}")

                commands.append(specific_cmd)

            if method == "natural-compression":
              if all("--wandb-name=" not in part for part in command):
                command.append(f"--wandb-name={build_wandb_name(command)}")

              commands.append(command)

  return commands


def create_jobs() -> List[str]:
  jobs = []

  cpu = 12
  memory = "32Gi"
  gpu = 1

  if gpu == 0:
    priority = "cpu-normal-batch"
  else:
    priority = "normal-batch"  # Options are: "low-batch", "normal-batch", "high-batch"

  commands = build_commands()

  for command in commands:
    job_name = build_wandb_name(command)
    job = JOB_TEMPLATE.format(
      NAME=job_name,
      COMMAND=command,
      OMP_NUM_THREADS=f"\"{cpu}\"",
      CPU=cpu,
      MEMORY=f"\"{memory}\"",
      GPU=gpu,
      PRIORITY=priority,
    )
    jobs.append(job)

  return jobs


def build_wandb_name(command: List[str]):
  if any("--wandb-name=" in part for part in command):
    for part in command:
      if "--wandb-name=" in part:
        return part.split("=")[1]

  # Use a set of important arguments for our experiment to build the wandb name.
  # Each argument will be separated by a dash. We also define an alias for each argument so that the name is more readable.
  important_args_aliases = {
    "-i": "case",
    "residual-stream-compression-size": "size",
    # "seed": "seed",
    # "ae-epochs": "ae-epochs",
    # "freeze-ae-weights": "frozen",
    # "ae-training-epochs-gap": "ae-gap",
    # "ae-desired-test-mse": "ae-mse",
    # "lr-patience": "lr-patience",
  }
  important_args = important_args_aliases.keys()
  wandb_name = ""

  # wandb_name += command[3] + "-"  # training method

  for arg in important_args:
    for part in command:
      if arg in part:
        alias = important_args_aliases[arg]
        if "=" in part:
          arg_value = part.split("=")[1].replace("_", "-")
          wandb_name += f"{alias}-{arg_value}-"
        else:
          wandb_name += f"{alias}-"

        break

  # remove last dash from wandb_name
  wandb_name = wandb_name[:-1]

  assert wandb_name != "", f"wandb_name is empty. command: {command}"

  return wandb_name


def launch_kubernetes_jobs():
  jobs = create_jobs()
  yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

  print(yamls_for_all_jobs)
  if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
    subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=yamls_for_all_jobs.encode())
    print(f"Jobs launched.")


def print_commands():
  commands = build_commands()
  for command in commands:
    job_name = build_wandb_name(command)
    print(f"Job: {job_name}")
    print(f"Command: {' '.join(command)}")
    print()


if __name__ == "__main__":
  launch_kubernetes_jobs()
  print_commands()
