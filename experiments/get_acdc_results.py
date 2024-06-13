#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb
from circuits_benchmark.utils.iit.best_weights import get_best_weight

cases = all_working_cases
weights = []
for case in cases:
    weights.append(get_best_weight(case))

def clean_wandb():
    print()
    try:
        print("Deleting acdc runs.")
        api = wandb.Api()
        project = "circuit_discovery"
        group = "acdc"
        runs = api.runs(f"{project}")
        # clean all runs in the group
        for run in runs:
            if (
                group in run.group
                and any(str(case) in run.group for case in cases)
                and (get_best_weight(case) in run.group or "tracr" in run.group)
            ):
                print(f"Deleting run {run.name}, {run.group}")
                run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")
    print()
    try:
        print("Deleting node_realism runs.")
        project = "node_realism"
        runs = api.runs(f"{project}")
        # clean all runs in the group
        for run in runs:
            if (
                "acdc" in run.group
                and any(str(case) in run.group for case in cases)
                and (get_best_weight(case) in run.group or "tracr" in run.group)
            ):
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

    acdc_command_template = """python main.py eval iit_acdc -i {} -w {} -t {} -wandb --load-from-wandb --abs-value-threshold"""
    circuit_score_command_template = """python main.py eval node_realism -i {} --mean --relative 1 -w {} -t {} --use-wandb --load-from-wandb"""
    commands = []
    for case in cases:
        commands_to_run = []
        tracr_commands_to_run = []
        weight = get_best_weight(case)
        for threshold in thresholds:
            acdc_command = acdc_command_template.format(case, weight, threshold)
            circuit_score_command = circuit_score_command_template.format(
                case, weight, threshold
            )
            commands_to_run.append(acdc_command)
            commands_to_run.append(circuit_score_command)

            tracr_command = acdc_command_template.format(case, "tracr", threshold)
            tracr_circuit_score_command = (
                circuit_score_command_template.format(case, "tracr", threshold)
                + " --tracr"
            )
            tracr_commands_to_run.append(tracr_command)
            tracr_commands_to_run.append(tracr_circuit_score_command)

        command = ["bash", "-c", " && ".join(commands_to_run)]
        tracr_command = ["bash", "-c", " && ".join(tracr_commands_to_run)]
        commands.append(command)
        commands.append(tracr_command)
    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    for arg in sys.argv:
        if arg in ["-d", "--dry-run"]:
            sys.exit(0)
        if arg in ["-l", "--local"]:
            print("Running locally.")
            clean_wandb()
            run_commands(build_commands())
            sys.exit(0)
        if arg in ["-c", "--clean"]:
            clean_wandb()
            sys.exit(0)
    clean_wandb()
    launch_kubernetes_jobs(build_commands, memory="14Gi", priority="high-batch")
