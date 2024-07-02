import traceback

from circuits_benchmark.commands.algorithms import acdc, eap, sp
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("run")
  run_subparsers = run_parser.add_subparsers(dest="algorithm")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  acdc.ACDCRunner.setup_subparser(run_subparsers)
  sp.SPRunner.setup_subparser(run_subparsers)
  eap.EAPRunner.setup_subparser(run_subparsers)


def run(args):
  for case in get_cases(args):
    print(f"\nRunning {args.algorithm} on {case}")

    try:
      if args.algorithm == "acdc":
        acdc.ACDCRunner(case, args).run_using_model_loader_from_args()
      if args.algorithm == "sp":
        sp.SPRunner(case, args).run_using_model_loader_from_args()
      if args.algorithm == "eap":
        eap.EAPRunner(case, args).run_using_model_loader_from_args()
    except Exception as e:
      print(f" >>> Failed to run {args.algorithm} on {case}:")
      traceback.print_exc()
      continue