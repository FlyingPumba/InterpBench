import wandb
import sys
import os

if __name__ == "__main__":
  if len(sys.argv) != 3:
    raise ValueError("Usage: python download_wandb_artifact.py <project_name> <artifact_name>")

  output_dir = "./artifacts"
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # first argument is project name, second is run id, third is artifact name
  project_name = sys.argv[1]
  artifact_name = sys.argv[2]

  api = wandb.Api()
  artifact = api.artifact(f"iarcuschin/{project_name}/{artifact_name}:latest")
  artifact.download(root=output_dir)