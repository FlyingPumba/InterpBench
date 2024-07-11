from typing import List, Callable
import sys
from .kubecalls import launch_kubernetes_jobs, print_commands
from .localcalls import run_commands

def main(
    build_commands: Callable[[], List[str]], 
    clean_wandb: Callable[[], None],
    priority: str = "normal-batch", # or "high-batch"
):
    for arg in sys.argv:
        if arg in ["-d", "--dry-run"]:
            print_commands(build_commands)
            sys.exit(0)
        if arg in ["-dc", "--dry-clean"]:
            clean_wandb(dry_run=True)
            sys.exit(0)
        if arg in ["-l", "--local"]:
            print("Running locally.")
            clean_wandb(dry_run=False)
            run_commands(build_commands())
            sys.exit(0)
        if arg in ["-c", "--clean"]:
            clean_wandb(dry_run=False)
            sys.exit(0)
        if arg in ["-r", "--run"]:
            launch_kubernetes_jobs(build_commands, memory="8Gi", priority=priority)
            sys.exit(0)
    clean_wandb(dry_run=False)
    launch_kubernetes_jobs(build_commands, memory="8Gi", priority=priority)