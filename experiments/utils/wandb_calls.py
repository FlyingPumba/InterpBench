from tqdm import tqdm
import wandb


def get_runs_with_substr(
    project: str,
    name_substr: str = "",
    group_substr: str = "",
):
    api = wandb.Api()
    runs = api.runs(f"{project}")

    for run in tqdm(runs):
        if name_substr in run.name and group_substr in run.group:
            yield run
