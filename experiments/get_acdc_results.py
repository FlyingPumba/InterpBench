#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb 

def clean_wandb():
    try:
        print("Deleting acdc runs.")
        api = wandb.Api()
        project = 'circuit_discovery'
        group = 'acdc'
        runs = api.runs(f'{project}', filters={"group": f'{group}'})
        # clean all runs in the group
        for run in runs:
            run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")
    try:
        print("Deleting node_realism runs.")
        project = 'node_realism'
        runs = api.runs(f'{project}')
        # clean all runs in the group
        for run in runs:
            if 'acdc' in run.group:
                run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")

def build_commands():
    cases = working_cases
    iit = 1.0
    strict = 0.4
    behavior = 1.0
    weight = int(strict * 1000 + behavior * 100 + iit * 10)
    thresholds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 
              0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 10.0, 20.0, 50.0, 100.0]

    acdc_command_template = """python main.py eval iit_acdc -i {} -w {} -t {} -wandb --load-from-wandb"""
    circuit_score_command_template = """python main.py eval node_realism -i {} --mean --relative 1 -w {} -t {} --use-wandb --load-from-wandb"""
    commands = []
    for case in cases:
        commands_to_run = []
        tracr_commands_to_run = []
        for threshold in thresholds:
            acdc_command = acdc_command_template.format(case, weight, threshold)
            circuit_score_command = circuit_score_command_template.format(case, weight, threshold)
            commands_to_run.append(acdc_command)
            commands_to_run.append(circuit_score_command)

            tracr_command = acdc_command_template.format(case, "tracr", threshold)
            tracr_circuit_score_command = circuit_score_command_template.format(case, "tracr", threshold) + " --tracr"
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
    launch_kubernetes_jobs(
        build_commands, memory="14Gi", priority="high-batch"
    )