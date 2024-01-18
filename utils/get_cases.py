import glob
import os


BENCHMARK_DIR = "benchmark"

def get_cases_files(args):
  if args is not None and args.indices is not None:
    # convert index to 5 digits
    files = [f"{BENCHMARK_DIR}/case-{int(index):05d}/rasp.py" for index in args.indices.split(",")]

    # Check that all the files exist.
    for file_path in files:
      if not os.path.exists(file_path):
        raise ValueError(f"Case with path {file_path} does not exist.")
  else:
    files = sorted(glob.glob(os.path.join(BENCHMARK_DIR, "case-*", "rasp.py")))

  return files