#!/usr/bin/env python3
import sys
from utils import *
from utils.wandb_calls import get_runs_with_substr


b = [0.5, 1]
iit = [0.5, 1, 1.2]
s = [0.4, 0.8, 1, 1.4, 1.8]
lr = 1e-3
batch = 1024
cases = working_cases

def clean_runs():
    pass

def build_commands():
    command_template = """python main.py train ioi --epochs 2000 -iit {} -s {} -b {} --use-wandb --wandb-suffix strict_{} --save-model-wandb --lr {} --batch {}"""

    commands = []
    
    for case in cases:
        for iit_ in iit:
            for s_ in s:
                for b_ in b:
                    suffix = f"case_{case}_iit_{iit_}_s_{s_}_b_{b_}"
                    command = command_template.format(case, iit_, s_, b_, suffix, lr, batch).split()
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
