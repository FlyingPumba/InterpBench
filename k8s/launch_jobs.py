#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path
from typing import List

JOB_TEMPLATE_PATH = Path(__file__).parent / "runner.yaml"
with JOB_TEMPLATE_PATH.open() as f:
  JOB_TEMPLATE = f.read()


# ae_command = [".venv/bin/python", "main.py",
#               "train", "autoencoder",
#               "-i=3",
#               f"--residual-stream-compression-size={compression_size}",
#               "--epochs=9000",
#               f"--seed={seed}",
#               f"--lr-start=0.01",
#               f"--wandb-name=ae-seed-{seed}-size-{compression_size}",
#               "--wandb-project=aes-for-non-linear-compression-with-frozen-autoencoder-variance-to-sizes-and-seed"]

# command = [".venv/bin/python", "main.py",
#            "train", "non-linear-compression",
#            "-i=3",
#            f"--residual-stream-compression-size={compression_size}",
#            f"--seed={seed}",
#            # "--train-data-size=256",
#            # "--test-data-ratio=0.3",
#            # "--batch-size=2048",
#            "--epochs=25000",
#            "--freeze-ae-weights",
#            f"--lr-start={lr_start}",
#            f"--ae-path=results/case-3-resid-{compression_size}-autoencoder-weights.pt",
#            f"--wandb-name=seed-{seed}-size-{compression_size}",
#            "--wandb-project=non-linear-compression-with-frozen-autoencoder-variance-to-sizes-and-seed"]

# join the commands using && and wrap them in bash -c "..."
# command = ["bash", "-c", f"{' '.join(ae_command)} && {' '.join(command)}"]

def build_commands_and_jobs_names():
  # training_methods = ["linear-compression", "non-linear-compression", "natural-compression"]
  training_methods = ["autoencoder"]
  cases = [48]
  compression_sizes = list(range(1, 90, 20))
  seeds = [52,53,54]
  lr_starts = [0.001]
  train_data_sizes = [1000]
  test_data_ratios = [0.3]
  batch_sizes = [2048]

  commands_and_job_names = []

  for method in training_methods:
    for case in cases:
      for compression_size in compression_sizes:
        for seed in seeds:
          for lr_start in lr_starts:
            for train_data_size in train_data_sizes:
              for test_data_ratio in test_data_ratios:
                for batch_size in batch_sizes:

                  epochs = 10000

                  method_name = method[:-len("-compression")] if method.endswith("-compression") else method

                  wandb_project = f"autoencoder-wide-vs-narrow"
                  wandb_name = f"narrow-seed-{seed}-size-{compression_size}"

                  # wandb_project = f"compression-comparison"
                  # wandb_name = f"{method_name}-narrow-frozen-ae"

                  job_name = (f"{method_name}-compression-{compression_size}-"
                              f"case-{case}-"
                              f"seed-{seed}-"
                              f"lr-{str(lr_start).replace('.', '-')}-narrow")

                  command = [".venv/bin/python", "main.py",
                             "train", method,
                             f"-i={case}",
                             f"--residual-stream-compression-size={compression_size}",
                             f"--seed={seed}",
                             f"--train-data-size={train_data_size}",
                             f"--test-data-ratio={test_data_ratio}",
                             f"--batch-size={batch_size}",
                             f"--epochs={epochs}",
                             f"--lr-start={lr_start}",
                             "--early-stop-test-accuracy=0.97",
                             f"--wandb-name={wandb_name}",
                             f"--wandb-project={wandb_project}"]

                  if method == "linear-compression":
                    command.append("--linear-compression-initialization=linear")

                  if method == "non-linear-compression":
                    command.append("--freeze-ae-weights")
                    command.append("--ae-first-hidden-layer-shape=narrow")

                  if method == "autoencoder":
                    command.append("--ae-layers=2")
                    command.append("--ae-first-hidden-layer-shape=narrow")

                  commands_and_job_names.append((command, job_name))

  return commands_and_job_names


def create_jobs() -> List[str]:
  jobs = []
  priority = "normal-batch"  # Options are: "low-batch", "normal-batch", "high-batch"

  cpu = 4
  memory = "16Gi"
  gpu = 0

  commands_and_jobs_names = build_commands_and_jobs_names()

  for command, job_name in commands_and_jobs_names:
    job = JOB_TEMPLATE.format(
      NAME=job_name,
      COMMAND=command,
      OMP_NUM_THREADS=f"\"{cpu}\"",
      CPU=cpu,
      MEMORY=f"\"{memory}\"",
      GPU=gpu,
      PRIORITY=priority,
    )
    jobs.append(job)

  return jobs


def launch_kubernetes_jobs():
  jobs = create_jobs()
  yamls_for_all_jobs = "\n\n---\n\n".join(jobs)

  print(yamls_for_all_jobs)
  if not any(s in sys.argv for s in ["--dryrun", "--dry-run", "-d"]):
    subprocess.run(["kubectl", "create", "-f", "-"], check=True, input=yamls_for_all_jobs.encode())
    print(f"Jobs launched.")

def print_commands():
  commands_and_jobs_names = build_commands_and_jobs_names()
  for command, job_name in commands_and_jobs_names:
    print(f"Job: {job_name}")
    print(f"Command: {' '.join(command)}")
    print()


if __name__ == "__main__":
  launch_kubernetes_jobs()
  print_commands()