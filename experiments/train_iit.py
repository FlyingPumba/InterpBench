#!/usr/bin/env python3
import sys
from utils import *

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]


def build_commands():
    # case_instances = get_cases(indices=None)
    cases = [1, 3, 4, 13, 21, 24, 27, 32, 38, 8] # 6, 18, 12, 19
    # cases = []

    # for case in case_instances:
    #     cases.append(case.get_index())

    command_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit 1 -s 0.4 --use-wandb --wandb-suffix strict_{}"""

    commands = []
    for case in cases:
        command = command_template.format(case, case).split()
        commands.append(command)

    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    for arg in sys.argv:
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
    launch_kubernetes_jobs(
        build_commands, cpu=1, gpu=1, memory="4Gi", priority="high-batch"
    )