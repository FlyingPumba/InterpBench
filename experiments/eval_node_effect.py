#!/usr/bin/env python3
from utils import *

def build_commands():
    cases = [1, 3, 4, 13, 21, 24, 27, 32, 38, 8]
    iit = 1.0
    strict = 0.4
    behavior = 1.0

    weight = int(strict * 1000 + behavior * 100 + iit * 10)

    # command1
    command1_template = """python main.py train iit -i {} --epochs 2000 --device cpu -iit {} -s {} -b {} --use-wandb --wandb-suffix case-{}-strict-{}"""

    # command2
    command2_template = """python main.py eval iit -i {} -w {} --save-to-wandb"""

    # join the commands using && and wrap them in bash -c "..."
    # command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

    commands = []
    for case in cases:
        command1 = command1_template.format(case, iit, strict, behavior, case, weight).split()
        command2 = command2_template.format(case, weight).split()
        joined_command = ["bash", "-c", f"{' '.join(command1)} && {' '.join(command2)}"]

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