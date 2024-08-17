from kube_utils import main, get_runs_with_substr
from circuits_benchmark.utils.get_cases import get_names_of_working_cases

def clean_wandb(*args, **kwargs):
    print("Not implemented yet")
    return 

def build_commands():
    strict_weights = [0.01, 0.05, 0.1, 0.4, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    # tasks = get_names_of_working_cases() + ["24", "7", "12", "14", "28"]
    # # remove tasks containing string 'ioi'
    # tasks = [task for task in tasks if 'ioi' not in task]
    tasks = ["3", "4", "8", "21", "24", "12", "28", "14", "7"]
    command = """python run_hyperparam_sweep.py -i {} -s {} --use-wandb"""

    commands = []
    for task in tasks:
        for weight in strict_weights:
            commands.append(command.format(task, weight).split(" "))
    return commands


if __name__ == "__main__":
    main(build_commands, clean_wandb)
