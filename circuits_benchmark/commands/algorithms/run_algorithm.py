import traceback

from circuits_benchmark.commands.algorithms import acdc, legacy_acdc, eap, sp, edge_pruning
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.utils.ll_model_loader.ll_model_loader_factory import get_ll_model_loader_from_args


def setup_args_parser(subparsers):
    run_parser = subparsers.add_parser("run")
    run_subparsers = run_parser.add_subparsers(dest="algorithm")
    run_subparsers.required = True

    # Setup arguments for each algorithm
    legacy_acdc.LegacyACDCRunner.setup_subparser(run_subparsers)
    acdc.ACDCRunner.setup_subparser(run_subparsers)
    sp.SPRunner.setup_subparser(run_subparsers)
    eap.EAPRunner.setup_subparser(run_subparsers)
    edge_pruning.EdgePruningRunner.setup_subparser(run_subparsers)


def run(args):
    for case in get_cases(args):
        print(f"\nRunning {args.algorithm} on {case}")

        ll_model_loader = get_ll_model_loader_from_args(case, args)

        try:
            if args.algorithm == "legacy_acdc":
                legacy_acdc.LegacyACDCRunner(case, args=args).run_using_model_loader(ll_model_loader)
            elif args.algorithm == "acdc":
                acdc.ACDCRunner(case, args=args).run_using_model_loader(ll_model_loader)
            elif args.algorithm == "sp":
                sp.SPRunner(case, args=args).run_using_model_loader(ll_model_loader)
            elif args.algorithm == "eap":
                eap.EAPRunner(case, args=args).run_using_model_loader(ll_model_loader)
            elif args.algorithm == "edge_pruning":
                edge_pruning.EdgePruningRunner(case, args=args).run_using_model_loader(ll_model_loader)
            else:
                raise ValueError(f"Unknown algorithm: {args.algorithm}")
        except Exception as e:
            print(f" >>> Failed to run {args.algorithm} algorithm on case {case}:")
            traceback.print_exc()
            continue
