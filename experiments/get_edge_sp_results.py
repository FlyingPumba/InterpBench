#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb 

all_case_objs = get_cases()
is_categoricals = [
    case_obj.build_transformer_lens_model().is_categorical()
    for case_obj in all_case_objs
]
metrics = ["kl" if is_categorical else "l2" for is_categorical in is_categoricals]
group = "edge_sp"
cases = all_working_cases
iit = 1.0
strict = 0.4
behavior = 1.0
weight = int(strict * 1000 + behavior * 100 + iit * 10)


def clean_wandb():
    print()
    try:
        print("Deleting node sp runs.")
        api = wandb.Api()
        project = "circuit_discovery"
        runs = api.runs(f"{project}")
        # clean all runs in the group
        for run in runs:
            if (
                group in run.group
                and any(str(case) in run.group for case in cases)
                and (str(weight) in run.group or "tracr" in run.group)
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
                group in run.group
                and any("_" + str(case) + "_" in run.group for case in cases)
                and (str(weight) in run.group or "tracr" in run.group)
            ):
                print(f"Deleting run {run.name}, {run.group}")
                run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")

def build_commands():
    lambda_regs = [0.0, 
              1e-5, 1e-4, 1e-3, 1e-2, 
              0.025, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0, 10.0, 20.0, 50.0, 100.0]

    sp_command_template = """python main.py run sp -i {} --metric {} --torch-num-threads 4 --device cuda --lambda-reg {} --epochs 3000 --load-from-wandb -w {} --using-wandb --wandb-project circuit_discovery --wandb-group edge_sp_{}_{} --wandb-run-name {} --edgewise"""
    circuit_score_command_template = """python main.py eval node_realism -i {} --algorithm edge_sp --mean --relative 0 -w {} --lambda-reg {} --use-wandb --load-from-wandb"""
    commands = []
    for case in cases:
        metric = metrics[case]
        commands_to_run = []
        tracr_commands_to_run = []
        for lambda_reg in lambda_regs:
            acdc_command = sp_command_template.format(case, metric, lambda_reg, weight, case, weight, lambda_reg)
            circuit_score_command = circuit_score_command_template.format(case, weight, lambda_reg)
            commands_to_run.append(acdc_command)
            commands_to_run.append(circuit_score_command)

            tracr_command = sp_command_template.format(case, metric, lambda_reg, "tracr", case, "tracr", lambda_reg) + " --tracr"
            tracr_circuit_score_command = circuit_score_command_template.format(case, "tracr", lambda_reg) + " --tracr"
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
        build_commands, memory="12Gi", priority="high-batch"
    )