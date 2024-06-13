#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb
from circuits_benchmark.utils.iit.best_weights import get_best_weight

cases = all_working_cases
weights = ["best", "100"]
algorithm = "acdc"


def clean_wandb():
    print()
    api = wandb.Api()
    try:
        print("Deleting node_realism runs.")
        project = "node_realism"
        runs = api.runs(f"{project}")
        # clean all runs in the group
        for run in runs:
            if (
                algorithm in run.group
                and any(f"_{case}_" in run.group for case in cases)
            ):
                task = run.group.split("_")[1]
                if (any(weight in run.group for weight in weights) 
                    or ("tracr" in run.group)
                    or (get_best_weight(task) in run.group)):
                    print(f"Deleting run {run.name}, {run.group}")
                    run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")


def build_commands():

    thresholds = [
        0.0,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        0.02,
        0.025,
        0.05,
        0.1,
        0.2,
        0.5,
        0.8,
        1.0,
        10.0,
        20.0,
        50.0,
        100.0,
    ]
    circuit_score_command_template = """python main.py eval node_realism -i {} --mean --relative 1 -w {} -t {} --use-wandb --load-from-wandb --algorithm {}"""
    commands = []
    for case in cases:
        commands_to_run = []
        for threshold in thresholds:
            for weight in weights:
                if weight == "best":
                    weight = get_best_weight(case)
                if weight != "510":
                    circuit_score_command = circuit_score_command_template.format(
                        case, "510", threshold, algorithm
                    ).split()
                    commands.append(circuit_score_command)
                circuit_score_command = circuit_score_command_template.format(
                    case, weight, threshold, algorithm
                ).split()
                commands.append(circuit_score_command)
            tracr_command = (
                circuit_score_command_template.format(
                    case, "tracr", threshold, algorithm
                )
                + " --tracr"
            ).split()
            commands.append(tracr_command)
        # command = ["bash", "-c", " && ".join(commands_to_run)]
        # commands.append(command)
    return commands


if __name__ == "__main__":
    for arg in sys.argv:
        if arg in ["-d", "--dry-run"]:
            print_commands(build_commands)
            sys.exit(0)
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
            sys.exit(0)
        if arg in ["-c", "--clean"]:
            clean_wandb()
            sys.exit(0)
    clean_wandb()
    launch_kubernetes_jobs(build_commands, memory="14Gi", priority="high-batch")
