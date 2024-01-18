import glob
import os

from benchmark.benchmark_case import BenchmarkCase

BENCHMARK_DIR = "benchmark"

def get_cases(args):
  if args is not None and args.indices is not None:
    # convert index to 5 digits
    files = [f"../{BENCHMARK_DIR}/case-{int(index):05d}/rasp.py" for index in args.indices.split(",")]

    # Check that all the files exist.
    for file_path in files:
      if not os.path.exists(file_path):
        raise ValueError(f"Case with path {file_path} does not exist.")
  else:
    files = sorted(glob.glob(os.path.join("../", BENCHMARK_DIR, "case-*", "rasp.py")))

  # remove "../" prefix from files
  file_paths_from_root = [file_path[3:] for file_path in files]

  return [BenchmarkCase.get_instance_for_file_path(path) for path in file_paths_from_root]