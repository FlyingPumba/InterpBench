import wandb
from tqdm import tqdm

def get_runs_with_substr(name_has_prefix):
    api = wandb.Api()
    project = "iit"
    runs = api.runs(f"{project}")

    # clean all runs in the group
    for run in tqdm(runs):
        if name_has_prefix in run.name:
            yield run