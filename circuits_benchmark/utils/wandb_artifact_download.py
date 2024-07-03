import os
import shutil
from pathlib import Path
from typing import List

import wandb

from circuits_benchmark.utils.project_paths import get_default_output_dir

default_artifacts_output_dir = os.path.join(get_default_output_dir(), "artifacts")


def download_artifact(project_name: str,
                      artifact_name: str,
                      output_dir: str = default_artifacts_output_dir) -> List[Path]:
  if os.path.exists(output_dir):
    # remove dir to clean previous downloads
    shutil.rmtree(output_dir)

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  api = wandb.Api()
  artifact = api.artifact(f"{project_name}/{artifact_name}:latest")
  artifact.download(root=output_dir)

  # return the name of the downloaded files (path objects)
  return list(Path(output_dir).iterdir())
