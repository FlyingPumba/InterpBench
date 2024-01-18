import traceback

from utils.get_cases import get_cases_files


def run(args):
  for file_path in get_cases_files(args):
    try:
      print(f"\nRunning {args.algorithm} on {file_path}")
    except Exception as e:
      print(f" >>> Failed to run on {file_path}:")
      traceback.print_exc()
      continue