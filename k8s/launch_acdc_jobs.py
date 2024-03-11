#!/usr/bin/env python3
import subprocess
import sys
from itertools import product
from math import ceil
from pathlib import Path
from typing import List

import numpy as np

from circuits_benchmark.utils.get_cases import get_cases

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
  JOB_TEMPLATE = f.read()

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands():
  case_instances = get_cases(indices=None)

  cases = []
  compression_sizes_by_case = {}
  for case in case_instances:
    original_resid_size = case.get_tl_model().cfg.d_model
    compression_sizes_by_case[case.get_index()] = [ceil(original_resid_size / 3)]
    cases.append(case.get_index())

  checkpoint_types = ["tracr", "linear-compression", "non-linear-compression", "natural-compression"]
  thresholds = 10**np.linspace(0, -8, 50)

  commands = []
  for case in cases:
    for compression_size in compression_sizes_by_case[case]:
      for checkpoint_type in checkpoint_types:
        for threshold in thresholds:

          wandb_project = f"acdc-experiment"

          if checkpoint_type == "non-linear-compression":
            artifact_name = f"case-{case}-resid-{compression_size}-non-linearly-compressed-tracr-transformer"
          elif checkpoint_type == "linear-compression":
            artifact_name = f"case-{case}-resid-{compression_size}-linearly-compressed-tracr-transformer"
          elif checkpoint_type == "natural-compression":
            artifact_name = f"case-{case}-resid-{compression_size}-naturally-compressed-tracr-transformer"
          elif checkpoint_type == "tracr":
            pass
          else:
            raise ValueError(f"Invalid checkpoint_type: {checkpoint_type}")

          threshold = f"{threshold:.3e}"

          command = [".venv/bin/python", "main.py",
                     "run",
                     "acdc",
                     f"-i={case}",
                     "--metric=l2",
                     f"--threshold={threshold}",
                     "--using-wandb",
                     "--wandb-entity-name=iarcuschin",
                     f"--wandb-project-name={wandb_project}",]

          if checkpoint_type != "tracr":
            command.append(f"--wandb-checkpoint-project-name=compress-all-cases")
            command.append(f"--wandb-checkpoint-artifact-name={artifact_name}")
            command.append(f"--wandb-checkpoint-type={checkpoint_type}")

          command.append(f"--wandb-run-name={build_wandb_name(command)}")
          commands.append(command)

  return commands


def create_jobs() -> List[str]:
  jobs = []
  priority = "normal-batch"  # Options are: "low-batch", "normal-batch", "high-batch"

  cpu = 4
  memory = "16Gi"
  gpu = 0

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
  # Use a set of important arguments for our experiment to build the wandb name.
  # Each argument will be separated by a dash. We also define an alias for each argument so that the name is more readable.
  important_args_aliases = {
    "-i": "case",
    "residual-stream-compression-size": "size",
    "threshold": "",
    "wandb-checkpoint-type": "",
    # "seed": "seed",
    # "ae-epochs": "ae-epochs",
    # "freeze-ae-weights": "frozen",
    # "ae-training-epochs-gap": "ae-gap",
    # "ae-desired-test-mse": "ae-mse",
    # "lr-patience": "lr-patience",
  }
  important_args = important_args_aliases.keys()
  wandb_name = ""

  wandb_name += command[3] + "-"  # training method

  for arg in important_args:
    found = False
    for part in command:
      if arg in part:
        found = True

        alias = important_args_aliases[arg]
        if alias != "":
          suffix = f"{alias}-"
        else:
          suffix = ""

        if "=" in part:
          arg_value = part.split("=")[1].replace(".", "-").replace("+", "-")
          wandb_name += f"{suffix}{arg_value}-"
        else:
          wandb_name += f"{suffix}"

        break

    if not found and arg == "wandb-checkpoint-type":
      wandb_name += "tracr-"

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
