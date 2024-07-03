import random

import numpy as np
import torch as t

from circuits_benchmark.commands.train.compression import linear_compression, \
  non_linear_compression
from circuits_benchmark.commands.train.compression.linear_compression import train_linear_compression
from circuits_benchmark.commands.train.compression.non_linear_compression import train_non_linear_compression
from circuits_benchmark.commands.train.iit import iit_train
from circuits_benchmark.utils.get_cases import get_cases


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("train")
  run_subparsers = run_parser.add_subparsers(dest="type")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  linear_compression.setup_args_parser(run_subparsers)
  non_linear_compression.setup_args_parser(run_subparsers)
  iit_train.setup_args_parser(run_subparsers)


def run(args):
  training_type = args.type

  cases = get_cases(args)
  assert len(cases) > 0, "No cases found"

  for case in cases:
    print(f"\nRunning training {training_type} on {case}")

    # Set numpy, torch and ptyhon seed
    seed = args.seed
    assert seed is not None, "Seed is always required"
    np.random.seed(args.seed)
    t.manual_seed(seed)
    random.seed(seed)

    if training_type == "linear-compression":
      train_linear_compression(case, args)
    elif training_type == "non-linear-compression":
      train_non_linear_compression(case, args)
    elif training_type == "iit":
      iit_train.run_iit_train(case, args)
    else:
      raise ValueError(f"Unknown training: {training_type}")
