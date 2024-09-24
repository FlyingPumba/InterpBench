import wandb


def load_model_from_wandb(
    case_index,
    weights: str = "510",
    output_dir: str = "./results",
    return_file_without_downloading: bool = False,
    same_size: bool = False,
    wandb_project: str | None = None,
    wandb_name: str | None = None
):
    api = wandb.Api()

    project = f"iit_models{'_same_size' if same_size else ''}"
    name = f"case_{case_index}_weight_{weights}"
    if wandb_project is not None:
        project = wandb_project
    if wandb_name is not None:
        name = wandb_name

    model_file_name = f"ll_model_{weights}.pt"
    cfg_file_name = f"ll_model_cfg_{weights}.pkl"
    runs = api.runs(project)
    for run in runs:
        if run.name != name:
            continue
        files = run.files()
        model_file = None
        cfg_file = None
        for file in files:
            if model_file_name in file.name:
                if not return_file_without_downloading:
                    file.download(replace=True, root=output_dir)
                model_file = file
            elif cfg_file_name in file.name:
                if not return_file_without_downloading:
                    file.download(replace=True, root=output_dir)
                cfg_file = file
        if model_file and cfg_file:
            return model_file, cfg_file
    raise FileNotFoundError(
        f"Could not find files {model_file_name} and {cfg_file_name} in run {name}"
    )


def load_circuit_from_wandb(
    case_index,
    algorithm: str,
    hyperparam: str,
    weights: str = "510",
    output_dir: str = "./results",
    return_file_without_downloading: bool = False,
    same_size: bool = False,
):
    api = wandb.Api()
    project = f"circuit_discovery{'_same_size' if same_size else ''}"
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
