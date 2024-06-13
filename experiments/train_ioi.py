#!/usr/bin/env python3
import sys
from utils import *
from utils.wandb_calls import get_runs_with_substr


b = [1]
iit = [1]
s = [0.4, 0.8, 1, 1.5, 2.0, 3.0, 4.0]
lr = 1e-3
batch = 1024

def clean_runs():
    runs = get_runs_with_substr("ioi", project="iit_models")
    for run in runs:
        print(f"Deleting run {run.name}")
        run.delete(delete_artifacts=True)

def build_commands():
    command_template = """python main.py train ioi --epochs 2000 -iit {} -s {} -b {} --use-wandb --wandb-suffix strict_{} --save-to-wandb --lr {} --batch {}"""
    commands = []
    
    for iit_ in iit:
        for s_ in s:
            for b_ in b:
                weight = int(1000*s_ + 100*b_ + 10*iit_)
                next_token_str = "_next_token" if b_ == 1 else ""
                suffix = f"ioi{next_token_str}_{weight}"
                command = command_template.format(iit_, s_, b_, suffix, lr, batch).split()
                command_next_token = command + ["--next-token"]
                command_include_mlp = command + ["--include-mlp"]
                command_include_mlp_next_token = command + ["--include-mlp", "--next-token"]
                commands += [command, command_next_token, command_include_mlp, command_include_mlp_next_token]

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
