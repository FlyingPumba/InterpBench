#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb 

iit = 1.0
strict = 0.0
behavior = 1.0
weight = int(strict * 1000 + behavior * 100 + iit * 10)
cases = working_cases

def clean_wandb():
    try:
        print("Deleting node_effect runs.")
        api = wandb.Api()
        project = 'node_effect'
        runs = api.runs(f'{project}')
        # clean all runs in the group
        for case in cases:
            for run in runs:
                if str(case) in run.name and str(weight) in run.name:
                    run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")

def build_commands():
    
    all_case_objs = get_cases()

    command_template = """python main.py eval iit -i {} -w {} --save-to-wandb --categorical-metric {} --load-from-wandb"""

    commands = []
    for case in cases:
        case_obj = all_case_objs[case-1]
        is_categorical = case_obj.build_transformer_lens_model().is_categorical()
        cat_mets = ["accuracy", "kl_div", "kl_div_self"] if is_categorical else ["accuracy"]
        for cat_met in cat_mets:
            command = command_template.format(case, weight, cat_met).split()
            commands.append(command)
    
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