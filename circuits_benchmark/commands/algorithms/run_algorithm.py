import traceback

from circuits_benchmark.commands.algorithms import acdc, eap
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.algorithms import sp

def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("run")
  run_subparsers = run_parser.add_subparsers(dest="algorithm")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  acdc.setup_args_parser(run_subparsers)
  sp.setup_args_parser(run_subparsers)
  eap.EAPRunner.setup_subparser(run_subparsers)


def run(args):
  for case in get_cases(args):
    print(f"\nRunning {args.algorithm} on {case}")

    try:
      if args.algorithm == "acdc":
        acdc.run_acdc(case, args)
      if args.algorithm == "sp":
        sp.run_sp(case, args)
      if args.algorithm == "eap":
        eap_runner = eap.EAPRunner(case, args)
        eap_runner.run_on_tracr_model()
    except Exception as e:
      print(f" >>> Failed to run {args.algorithm} on {case}:")
      traceback.print_exc()
      continue