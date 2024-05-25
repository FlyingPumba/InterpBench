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


def build_commands():
    print("Getting runs...")
    runs = get_runs_with_substr("supervised")
    print("Deleting runs...")
    for run in tqdm(runs):
        run.delete(delete_artifacts=True)
    case_instances = get_cases(indices=None)
    cases = []

    for case in case_instances:
        cases.append(case.get_index())

    command_template = """python main.py train iit -i {} --epochs 500 --device cpu -iit 0 -s 0 --use-wandb --wandb-suffix supervised_{}"""

    commands = []
    for case in cases:
        command = command_template.format(case, case).split()
        commands.append(command)

    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    launch_kubernetes_jobs(
        build_commands, cpu=1, gpu=1, memory="8Gi", priority="high-batch"
    )
