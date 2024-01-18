import traceback

from commands import acdc
from utils.get_cases import get_cases_files


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("run")
  run_subparsers = run_parser.add_subparsers(dest="algorithm")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  acdc.setup_args_parser(run_subparsers)


def run(args):
  for file_path in get_cases_files(args):
    print(f"\nRunning {args.algorithm} on {file_path}")

    try:
      if args.algorithm == "acdc":
        acdc.run_acdc(file_path, args)

    except Exception as e:
      print(f" >>> Failed to run {args.algorithm} on {file_path}:")
      traceback.print_exc()
      continue