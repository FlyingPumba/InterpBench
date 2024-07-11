from typing import List, Callable
import sys
from .kubecalls import launch_kubernetes_jobs, print_commands
from .localcalls import run_commands

def main(build_commands: Callable[[], List[str]], clean_wandb: Callable[[], None]):
    print_commands(build_commands)
    for arg in sys.argv:
        if arg in ["-d", "--dry-run"]:
            sys.exit(0)
        if arg in ["-l", "--local"]:
            print("Running locally.")
            clean_wandb()
            run_commands(build_commands())
            sys.exit(0)
        if arg in ["-c", "--clean"]:
            clean_wandb()
            sys.exit(0)
    clean_wandb()
    launch_kubernetes_jobs(build_commands, memory="8Gi", priority="high-batch")