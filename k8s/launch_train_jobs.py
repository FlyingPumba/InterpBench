#!/usr/bin/env python3
import subprocess
import sys
from itertools import product
from math import ceil
from pathlib import Path
from typing import List

from circuits_benchmark.utils.get_cases import get_cases

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
  JOB_TEMPLATE = f.read()

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands():
  # training_methods = ["linear-compression", "non-linear-compression", "natural-compression", "autoencoder"]
  training_methods = ["linear-compression", "non-linear-compression"]
  case_instances = get_cases(indices=["1", "6", "7", "8", "10", "11", "12", "13", "14", "15", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33", "34", "35", "36", "37", "38"])

  cases = []
  compression_sizes_by_case = {}
  for case in case_instances:
    original_resid_size = case.get_tl_model().cfg.d_model
    compression_sizes_by_case[case.get_index()] = [ceil(original_resid_size * 2 / 3)]
    cases.append(case.get_index())

  seeds = [68]
  lr_starts = [0.001]
  epochs = 150 * 1000
  train_data_sizes = [1000]
  test_data_ratios = [0.3]
  batch_sizes = [2048]

  linear_compression_args = {
    "train-loss": ["intervention"],
    "linear-compression-initialization": ["linear"],  # ["orthogonal", "linear"],
  }

  non_linear_compression_args = {
    "train-loss": ["intervention"],
    "ae-layers": [2],
    "ae-first-hidden-layer-shape": ["wide"],  # ["narrow", "wide"],
    "ae-epochs": [100],
    "freeze-ae-weights": [False],
  }

  non_linear_compression_continuous_ae_training_args = {
    "ae-training-epochs-gap": [50],
    "ae-desired-test-mse": [1e-5]
  }

  autoencoder_args = {
    "ae-layers": [2],
    "ae-first-hidden-layer-shape": ["wide"],  # ["narrow", "wide"],
  }

  commands = []

  for method in training_methods:
    for case in cases:
      for compression_size in compression_sizes_by_case[case]:
        for seed in seeds:
          for lr_start in lr_starts:
            for train_data_size in train_data_sizes:
              for test_data_ratio in test_data_ratios:
                for batch_size in batch_sizes:

                  wandb_project = f"linear-vs-non-linear"

                  command = [".venv/bin/python", "main.py",
                             "train", method,
                             f"-i={case}",
                             f"--residual-stream-compression-size={compression_size}",
                             f"--seed={seed}",
                             f"--train-data-size={train_data_size}",
                             f"--test-data-ratio={test_data_ratio}",
                             f"--batch-size={batch_size}",
                             f"--epochs={epochs}",
                             f"--lr-start={lr_start}",
                             f"--lr-patience=2000",
                             # "--early-stop-test-accuracy=0.97",
                             "--resample-ablation-loss=True",
                             "--resample-ablation-data-size=1000",
                             "--resample-ablation-max-interventions=50",
                             "--resample-ablation-loss-epochs-gap=25",
                             f"--wandb-project={wandb_project}"]

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

                      specific_cmd.append(f"--wandb-name={build_wandb_name(specific_cmd)}")
                      commands.append(specific_cmd)

                  if method == "non-linear-compression":
                    # produce all combinations of args in non_linear_compression_args
                    arg_names = list(non_linear_compression_args.keys())
                    arg_values = list(non_linear_compression_args.values())
                    frozen_ae_weights_arg_idx = arg_names.index("freeze-ae-weights")
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

                      # If this is a non-frozen autoencoder training, add the autoencoder training command args
                      if not arg_values_combination[frozen_ae_weights_arg_idx]:
                        non_frozen_arg_names = list(non_linear_compression_continuous_ae_training_args.keys())
                        non_frozen_arg_values = list(non_linear_compression_continuous_ae_training_args.values())
                        for non_frozen_arg_values_combination in product(*non_frozen_arg_values):
                          more_specific_cmd = specific_cmd.copy()
                          for i, arg_name in enumerate(non_frozen_arg_names):
                            more_specific_cmd.append(f"--{arg_name}={non_frozen_arg_values_combination[i]}")

                          more_specific_cmd.append(f"--wandb-name={build_wandb_name(more_specific_cmd)}")
                          commands.append(more_specific_cmd)
                      else:
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

                      specific_cmd.append(f"--wandb-name={build_wandb_name(specific_cmd)}")
                      commands.append(specific_cmd)

                  if method == "natural-compression":
                    command.append(f"--wandb-name={build_wandb_name(command)}")
                    commands.append(command)

  return commands


def create_jobs() -> List[str]:
  jobs = []
  priority = "cpu-normal-batch"  # Options are: "low-batch", "normal-batch", "high-batch"

  cpu = 8
  memory = "4Gi"
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
    for part in command:
      if arg in part:
        alias = important_args_aliases[arg]
        if "=" in part:
          arg_value = part.split("=")[1]
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
