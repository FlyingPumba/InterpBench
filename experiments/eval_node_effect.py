#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb 


def build_commands():
    api = wandb.Api()
    project = 'iit'
    group = 'eval'
    runs = api.runs(f'{project}', filters={"group": group})

    # clean all runs in the group
    for run in runs:
        run.delete(delete_artifacts=True)

    cases = [1, 3, 4, 13, 21, 24, 27, 32, 38, 8]
    iit = 1.0
    strict = 0.4
    behavior = 1.0
    all_case_objs = get_cases()
    weight = int(strict * 1000 + behavior * 100 + iit * 10)

    # command1
    command1_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit {} -s {} -b {} --use-wandb --wandb-suffix case-{}-strict-{}"""

    # command2
    command2_template = """python main.py eval iit -i {} -w {} --save-to-wandb --categorical-metric {}"""

    # join the commands using && and wrap them in bash -c "..."
    # command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

    commands = []
    for case in cases:
        case_obj = all_case_objs[case-1]
        is_categorical = case_obj.build_transformer_lens_model().is_categorical()
        cat_mets = ["accuracy", "kl_div"] if is_categorical else ["kl_div"]
        for cat_met in cat_mets:
            command1 = command1_template.format(case, iit, strict, behavior, case, weight).split()
            command2 = command2_template.format(case, weight, cat_met).split()
            command3 = command2_template.format(case, "tracr", cat_met).split()
            joined_command = ["bash", "-c", f"{' '.join(command1)} && {' '.join(command2)} && {' '.join(command3)}"]

            commands.append(joined_command)
            
    
    return commands

if __name__ == "__main__":
    print_commands(build_commands)
    for arg in sys.argv:
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
    launch_kubernetes_jobs(
        build_commands, cpu=1, gpu=1, memory="12Gi", priority="high-batch"
    )