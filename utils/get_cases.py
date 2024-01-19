import glob
import os
from typing import List

from benchmark.benchmark_case import BenchmarkCase
from utils.relativize_path import relativize_path_to_project_root

BENCHMARK_DIR = "benchmark"

def get_cases(args) -> List[BenchmarkCase]:
  relative_benchmark_dir = relativize_path_to_project_root(BENCHMARK_DIR)
  if args is not None and args.indices is not None:
    # convert index to 5 digits
    files = [f"{relative_benchmark_dir}/case-{int(index):05d}/rasp.py" for index in args.indices.split(",")]

    # Check that all the files exist.
    for file_path in files:
      if not os.path.exists(file_path):
        raise ValueError(f"Case with path {file_path} does not exist.")
  else:
    files = sorted(glob.glob(os.path.join(relative_benchmark_dir, "case-*", "rasp.py")))

  # remove "../" prefix from files, as many times as needed
  for i in range(len(files)):
    while files[i][:3] == "../":
      files[i] = files[i][3:]

  return [BenchmarkCase.get_instance_for_file_path(path) for path in files]