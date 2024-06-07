import random

import numpy as np
import torch as t

from circuits_benchmark.commands.evaluation.iit import iit_eval, iit_acdc_eval, ioi_eval, ioi_acdc_eval
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.evaluation.realism import node_wise_ablation


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("eval")
  run_subparsers = run_parser.add_subparsers(dest="type")
  run_subparsers.required = True

  # Setup arguments for each evaluation type
  iit_eval.setup_args_parser(run_subparsers)
  iit_acdc_eval.setup_args_parser(run_subparsers)
  node_wise_ablation.setup_args_parser(run_subparsers)
  ioi_eval.setup_args_parser(run_subparsers)
  ioi_acdc_eval.setup_args_parser(run_subparsers)


def run(args):
  evaluation_type = args.type
  if evaluation_type == "ioi":
    ioi_eval.run_eval_ioi(args)
    return
  elif evaluation_type == "ioi_acdc":
    ioi_acdc_eval.run_ioi_acdc(args)
    return
  
  for case in get_cases(args):
    print(f"\nRunning evaluation {args.type} on {case}")

    # Set numpy, torch and ptyhon seed
    seed = args.seed
    assert seed is not None, "Seed is always required"
    np.random.seed(args.seed)
    t.manual_seed(seed)
    random.seed(seed)

    if evaluation_type == "iit":
      iit_eval.run_iit_eval(case, args)
    elif evaluation_type == "iit_acdc":
      iit_acdc_eval.run_acdc_eval(case, args)
    elif evaluation_type == "node_realism":
      node_wise_ablation.run_nodewise_ablation(case, args)
    else:
      raise ValueError(f"Unknown evaluation: {evaluation_type}")
