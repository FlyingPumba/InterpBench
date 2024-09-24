import random
import traceback

import numpy as np
import torch as t

from circuits_benchmark.commands.evaluation.iit import iit_eval
from circuits_benchmark.commands.evaluation.realism import node_wise_ablation, gt_circuit_node_wise_ablation
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
    run_parser = subparsers.add_parser("eval")
    run_subparsers = run_parser.add_subparsers(dest="type")
    run_subparsers.required = True

    # Setup arguments for each evaluation type
    iit_eval.setup_args_parser(run_subparsers)
    node_wise_ablation.setup_args_parser(run_subparsers)
    gt_circuit_node_wise_ablation.setup_args_parser(run_subparsers)


def run(args):
    evaluation_type = args.type
    for case in get_cases(args):
        print(f"\nRunning evaluation {evaluation_type} on {case}")

        # Set numpy, torch and ptyhon seed
        seed = args.seed
        assert seed is not None, "Seed is always required"
        np.random.seed(args.seed)
        t.manual_seed(seed)
        random.seed(seed)

        try:
            if evaluation_type == "iit":
                iit_eval.run_iit_eval(case, args)
            elif evaluation_type == "node_realism":
                node_wise_ablation.run_nodewise_ablation(case, args)
            elif evaluation_type == "gt_node_realism":
                gt_circuit_node_wise_ablation.run_nodewise_ablation(case, args)
            else:
                raise ValueError(f"Unknown evaluation: {evaluation_type}")
        except Exception as e:
            print(f" >>> Failed to run {evaluation_type} evaluation on case {case}:")
            traceback.print_exc()
            continue
