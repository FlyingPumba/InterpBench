from utils import *
import wandb
working = [11, 13, 18, 19, 20, 21, 26, 29, 3, 33, 34, 35, 36, 37, 4, 8]

def clean_wandb():
    print()
    project_names = ["circuit_discovery_same_size", "node_realism_same_size", "iit_models_same_size"]
    api = wandb.Api()
    def clean_project(name: str):
        try:
            print(f"Deleting {name} runs.")
            project = name
            runs = api.runs(f"{project}")
            for run in runs:
                if any(str(case) in run.group for case in working):
                    print(f"Deleting run {run.name}, {run.group}")
                    run.delete(delete_artifacts=True)
        except Exception as e:
            print("No runs found to delete.")
    for project_name in project_names:
        clean_project(project_name)
    print()

def build_commands():
    command_template = """python run_same_size_realism.py -i {}"""
    commands = []
    for case in working:
        commands.append(command_template.format(case).split(" "))
    return commands

if __name__ == "__main__":
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
    launch_kubernetes_jobs(build_commands, memory="12Gi", priority="high-batch")
