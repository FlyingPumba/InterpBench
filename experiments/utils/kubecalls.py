from typing import List, Callable
from pathlib import Path
import subprocess
import sys
from .aliases import important_args_aliases, replace_bad_things_in_arg

JOB_TEMPLATE_PATH = Path(__file__).parent.parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
    JOB_TEMPLATE = f.read()

def create_jobs(
    build_commands: Callable,
    memory: str = "32Gi",
    cpu: int = 8,
    gpu: int = 1,
    priority: str = "normal-batch",  # Options are: "low-batch", "normal-batch", "high-batch"
) -> List[str]:
    jobs = []

    commands = build_commands()
    print(len(commands), "commands found.")
    for command in commands:
        job_name = build_job_name(command)
        job = JOB_TEMPLATE.format(
            NAME=job_name,
            COMMAND=command,
            OMP_NUM_THREADS=f'"{cpu}"',
            CPU=cpu,
            MEMORY=f'"{memory}"',
            GPU=gpu,
            PRIORITY=priority,
        )
        jobs.append(job)
    print(len(jobs), "jobs created.")

    return jobs


def build_job_name(command: List[str]):
    # Use a set of important arguments for our experiment to build the wandb name.
    # Each argument will be separated by a dash. We also define an alias for each argument so that the name is more readable.
    important_args = important_args_aliases.keys()
    wandb_name = ""
    split_command = []
    for c in command:
        if ' ' in c:
            split_command.extend(c.split(' '))
        else:
            split_command.append(c)
    for arg in important_args:
        found = False
        alias = important_args_aliases[arg]
        for i, part in enumerate(split_command):
            if arg in part:
                found = True
                suffix = "" if alias == "" else f"{alias}-"
                if "=" in part:
                    arg_value = part.split("=")[1]
                else:
                    try:
                        arg_value = split_command[i + 1]
                    except IndexError:
                        arg_value = ""
                arg_value = replace_bad_things_in_arg(arg_value)
                wandb_name += f"{suffix}{arg_value}-"
                break
        if not found:
            wandb_name += f"{alias}-"
    # remove leading and trailing dashes
    wandb_name = wandb_name.strip("-")

    assert wandb_name != "", f"wandb_name is empty. command: {split_command}"

    return wandb_name


def launch_kubernetes_jobs(*args, **kwargs):
    jobs = create_jobs(*args, **kwargs)
    yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

    print(yamls_for_all_jobs.encode())

    if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
        subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=yamls_for_all_jobs.encode())
        print(f"Jobs launched.")


def print_commands(build_commands: Callable):
    commands = build_commands()
    for command in commands:
        job_name = build_job_name(command)
        print(f"Job: {job_name}")
        print(f"Command: {' '.join(command)}")
        print()
