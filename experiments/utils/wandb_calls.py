import wandb
from tqdm import tqdm

def get_runs_with_substr(name_has_prefix, project="iit"):
    api = wandb.Api()
    runs = api.runs(f"{project}")

    # clean all runs in the group
    for run in tqdm(runs):
        if name_has_prefix in run.name:
            yield run