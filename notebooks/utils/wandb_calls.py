import wandb
import concurrent.futures
from tqdm import tqdm


def fetch_files_from_runs(
    project: str,
    group: str,
    files_to_download: list[str],
    base_path="./results",
    state="finished",
    name_substr=None,
):
    api = wandb.Api()
    runs = api.runs(
        f"{project}", filters={"state": f"{state}"}
    )

    def download_files(run):
        if group not in run.group:
            return
        if (name_substr is not None) and (name_substr not in run.name):
            return
        for file in run.files():
            if any([file.name.endswith(ftd) for ftd in files_to_download]):
                file.download(replace=True, root=base_path)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(download_files, runs), total=len(runs)))
