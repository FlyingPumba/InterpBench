#!/usr/bin/env python3
import sys
from utils import *
from utils.wandb_calls import get_runs_with_substr
# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]
weight = "510"
s = "0.4"
iit = "1"
b = "1"
cases = working_cases
def clean_runs():
    runs = get_runs_with_substr(weight, project="iit_models")
    for run in runs:
        if any([f"case_{x}_" in run.name for x in cases]):
            print(f"Deleting run {run.name}")
            run.delete(delete_artifacts=True)

def build_commands(): 
    command_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit 1 -s 0.4 --use-wandb --wandb-suffix strict_{} --save-model-wandb"""

    commands = []
    for case in cases:
        command = command_template.format(case, case).split()
        commands.append(command)

    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    clean_runs()
    for arg in sys.argv:
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
    launch_kubernetes_jobs(
        build_commands, cpu=1, gpu=1, memory="12Gi", priority="high-batch"
    )