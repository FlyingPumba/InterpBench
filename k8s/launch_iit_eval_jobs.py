#!/usr/bin/env python3
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from circuits_benchmark.utils.get_cases import get_cases

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()


# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands():
    all_cases = get_cases()

    # load metadata json
    with open("/home/ivan/src/circuits-benchmark/metadata/benchmark_metadata.json", "r") as f:
        metadata = json.load(f)

    hf_cases = [d["case_id"] for d in metadata["cases"]]

    # take only the cases that are in the metadata
    cases = [case for case in all_cases if case.get_name() in hf_cases]
    seeds = [92]

    commands = []
    for case in cases:
        for seed in seeds:
            command = [
                ".venv/bin/python", "main.py",
                "eval", "iit",
                f"-i={case.get_name()}",
                "--interp-bench",
                f"--seed={seed}",
                f"--use-wandb",
                f"--wandb-project=iit-eval",
                f"--wandb-name=case-{case.get_name()}",
            ]

            if "ioi" in case.get_name():
                command.append("--include-mlp")

                if "next_token" in case.get_name():
                    command.append("--next-token")

            commands.append(command)

    return commands


def create_jobs() -> List[str]:
    jobs = []

    cpu = 4
    memory = "25Gi"
    gpu = 1

    if gpu == 0:
        priority = "cpu-normal-batch"
    else:
        priority = "normal-batch"  # Options are: "low-batch", "normal-batch", "high-batch"

    commands = build_commands()

    for command in commands:
        job_name = build_wandb_name(command)
        job = JOB_TEMPLATE.format(
            NAME=job_name,
            COMMAND=command,
            OMP_NUM_THREADS=f"\"{cpu}\"",
            CPU=cpu,
            MEMORY=f"\"{memory}\"",
            GPU=gpu,
            PRIORITY=priority,
        )
        jobs.append(job)

    return jobs


def build_wandb_name(command: List[str]):
    if any("--wandb-name=" in part for part in command):
        for part in command:
            if "--wandb-name=" in part:
                return part.split("=")[1].replace("_", "-")

    # Use a set of important arguments for our experiment to build the wandb name.
    # Each argument will be separated by a dash. We also define an alias for each argument so that the name is more readable.
    important_args_aliases = {
        "i": "case",
        "seed": "seed",
        "s": "s",
        "b": "b",
        "iit": "iit",
    }
    important_args = important_args_aliases.keys()
    wandb_name = ""

    # wandb_name += command[3] + "-"  # training method

    for arg in important_args:
        for part in command:
            if part.startswith(f"-{arg}") or part.startswith(f"--{arg}"):
                alias = important_args_aliases[arg]
                if "=" in part:
                    arg_value = part.split("=")[1].replace("_", "-")
                    wandb_name += f"{alias}-{arg_value}-"
                else:
                    wandb_name += f"{alias}-"

                break

    # remove last dash from wandb_name
    wandb_name = wandb_name[:-1]

    assert wandb_name != "", f"wandb_name is empty. command: {command}"

    return wandb_name


def launch_kubernetes_jobs():
    jobs = create_jobs()
    yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

    print(yamls_for_all_jobs)
    if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
        subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=yamls_for_all_jobs.encode())
        print(f"Jobs launched.")


def print_commands():
    commands = build_commands()
    for command in commands:
        job_name = build_wandb_name(command)
        print(f"Job: {job_name}")
        print(f"Command: {' '.join(command)}")
        print()


if __name__ == "__main__":
    launch_kubernetes_jobs()
    print_commands()
