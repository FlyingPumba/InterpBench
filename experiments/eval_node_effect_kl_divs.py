#!/usr/bin/env python3
from utils import *
from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.get_cases import get_cases
import wandb

iit = 1.0
strict = 0.4
behavior = 1.0
weight = int(strict * 1000 + behavior * 100 + iit * 10)
cases = working_cases
all_case_objs = get_cases()

cat_mets = ["kl_div_self", "kl_div"]

def clean_wandb():
    try:
        print("Deleting node_effect runs.")
        api = wandb.Api()
        project = "node_effect"
        runs = api.runs(f"{project}")
        # clean all runs in the group
        for case in cases:
            if not all_case_objs[case - 1].build_transformer_lens_model().is_categorical():
                continue
            for run in runs:
                if (
                    str(case) in run.name
                    and (str(weight) in run.name or "tracr" in run.name)
                    and any([cat_met in run.name for cat_met in cat_mets])
                ):
                    print(f"Deleting {run.name}")
                    run.delete(delete_artifacts=True)
    except Exception as e:
        print("No runs found to delete.")


def build_commands():
    command_template = """python main.py eval iit -i {} -w {} --save-to-wandb --categorical-metric {} --load-from-wandb"""

    commands = []
    for case in cases:
        case_obj = all_case_objs[case - 1]
        print(case_obj.get_index(), case - 1)
        is_categorical = (
            case_obj.build_transformer_lens_model().is_categorical()
        )
        if not is_categorical:
            continue
        for cat_met in cat_mets:
            command = command_template.format(case, weight, cat_met).split()
            commands.append(command)

        for cat_met in cat_mets:
            command_tracr = command_template.format(
                case, "tracr", cat_met
            ).split()
            commands.append(command_tracr)

    return commands


if __name__ == "__main__":
    print_commands(build_commands)
    clean_wandb()
    for arg in sys.argv:
        if arg in ["-l", "--local"]:
            print("Running locally.")
            run_commands(build_commands())
    launch_kubernetes_jobs(
        build_commands, cpu=1, gpu=1, memory="12Gi", priority="high-batch"
    )
