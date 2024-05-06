import random

import numpy as np
import torch as t

from circuits_benchmark.commands.evaluation.iit import iit_eval
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("eval")
  run_subparsers = run_parser.add_subparsers(dest="type")
  run_subparsers.required = True

  # Setup arguments for each evaluation type
  iit_eval.setup_args_parser(run_subparsers)


def run(args):
  for case in get_cases(args):
    evaluation_type = args.type
    print(f"\nRunning evaluation {args.type} on {case}")

    # Set numpy, torch and ptyhon seed
    seed = args.seed
    assert seed is not None, "Seed is always required"
    np.random.seed(args.seed)
    t.manual_seed(seed)
    random.seed(seed)

    if evaluation_type == "iit":
      iit_eval.run_iit_eval(case, args)
    else:
      raise ValueError(f"Unknown evaluation: {evaluation_type}")
