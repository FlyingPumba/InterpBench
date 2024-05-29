#!/usr/bin/env python3
from math import ceil
from pathlib import Path
from utils import *
from utils.wandb_calls import get_runs_with_substr
from tqdm import tqdm

from circuits_benchmark.utils.get_cases import get_cases

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]
cases = working_cases

def clean_runs():
    runs = get_runs_with_substr("supervised")
    for run in tqdm(runs):
        for case in cases:
            if str(case) in run.name:
                run.delete(delete_artifacts=True)
    model_runs = get_runs_with_substr("100", project="iit_models")
    for run in tqdm(model_runs):
        run.delete(delete_artifacts=True)

def build_commands():
    command_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit 0 -s 0 --use-wandb --wandb-suffix supervised_{} --save-model-wandb"""

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
