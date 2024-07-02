import traceback

from circuits_benchmark.commands.algorithms import acdc, eap, sp
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader_from_args


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

    ll_model_loader = get_ll_model_loader_from_args(case, args)

    try:
      if args.algorithm == "acdc":
        acdc.ACDCRunner(case, args).run_using_model_loader(ll_model_loader)
      if args.algorithm == "sp":
        sp.SPRunner(case, args).run_using_model_loader(ll_model_loader)
      if args.algorithm == "eap":
        eap.EAPRunner(case, args).run_using_model_loader(ll_model_loader)
    except Exception as e:
      print(f" >>> Failed to run {args.algorithm} on {case}:")
      traceback.print_exc()
      continue