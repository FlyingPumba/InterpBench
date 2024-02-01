import traceback

from commands.algorithms import acdc
from utils.get_cases import get_cases


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("run")
  run_subparsers = run_parser.add_subparsers(dest="algorithm")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  acdc.setup_args_parser(run_subparsers)


def run(args):
  for case in get_cases(args):
    print(f"\nRunning {args.algorithm} on {case}")

    try:
      if args.algorithm == "acdc":
        acdc.run_acdc(case, args)

    except Exception as e:
      print(f" >>> Failed to run {args.algorithm} on {case}:")
      traceback.print_exc()
      continue