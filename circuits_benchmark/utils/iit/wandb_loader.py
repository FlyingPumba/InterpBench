import wandb

def load_model_from_wandb(
        case_index,
        weights: str = "510",
        output_dir: str = "./results",
        return_file_without_downloading: bool = False
):
    api = wandb.Api()
    project = "iit_models"
    name = f"case_{case_index}_weight_{weights}"
    model_file_name = f"ll_model_{weights}.pt"
    runs = api.runs(project)
    for run in runs:
        if run.name != name:
            continue
        files = run.files()
        for file in files:
            if model_file_name in file.name:
                if not return_file_without_downloading:
                    file.download(replace=True, root=output_dir)
                return file
    raise FileNotFoundError(f"Could not find model file {model_file_name} in run {name}")


def load_circuit_from_wandb(
        case_index,
        algorithm: str,
        hyperparam: str,
        weights: str = "510",
        output_dir: str = "./results",
        return_file_without_downloading: bool = False
):
    api = wandb.Api()
    project = "circuit_discovery"
    group = f"{algorithm}_{case_index}_{weights}"
    runs = api.runs(project)
    for run in runs:
        if group not in run.group:
            continue
        if run.name != hyperparam:
            continue
        files = run.files()
        for file in files:
            if "result.pkl" in file.name:
                if not return_file_without_downloading:
                    file.download(replace=True, root=output_dir)
                return file
        raise FileNotFoundError(f"Could not find circuit file in run {hyperparam}")
    raise ValueError(f"Could not find run {hyperparam} in group {group}")