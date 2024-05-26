#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb 


def build_commands():
    try:
        api = wandb.Api()
        project = 'acdc'
        group = 'RQ3'
        runs = api.runs(f'{project}', filters={"group": group})
        # clean all runs in the group
        for run in runs:
            run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")

    cases = [1, 3, 4, 13, 21, 24, 27, 32, 38, 8, 19]
    iit = 1.0
    strict = 0.4
    behavior = 1.0
    weight = int(strict * 1000 + behavior * 100 + iit * 10)
    thresholds = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 
              0.025, 0.05, 0.1, 0.5, 0.8, 1.0, 10.0]

    train_command_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit {} -s {} -b {} --use-wandb --wandb-suffix case-{}-strict-{}"""
    acdc_command_template = """python main.py eval iit_acdc -i {} -w {} -t {} -wandb"""
    commands = []
    for case in cases:
        commands_to_run = []
        tracr_commands_to_run = []
        train_command = train_command_template.format(case, iit, strict, behavior, case, strict)
        commands_to_run.append(train_command)
        for threshold in thresholds:
            acdc_command = acdc_command_template.format(case, weight, threshold)
            tracr_command = acdc_command_template.format(case, "tracr", threshold)
            commands_to_run.append(acdc_command)
            tracr_commands_to_run.append(tracr_command)
        command = ["bash", "-c", " && ".join(commands_to_run)]
        tracr_command = ["bash", "-c", " && ".join(tracr_commands_to_run)]
        commands.append(command)
        commands.append(tracr_command)
    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    for arg in sys.argv:
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
    launch_kubernetes_jobs(build_commands)