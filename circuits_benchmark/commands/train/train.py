import random

import numpy as np
import torch as t

from circuits_benchmark.commands.train.compression import natural_compression, linear_compression, autoencoder, \
  non_linear_compression
from circuits_benchmark.commands.train.compression.autoencoder import train_autoencoder
from circuits_benchmark.commands.train.compression.linear_compression import train_linear_compression
from circuits_benchmark.commands.train.compression.non_linear_compression import train_non_linear_compression
from circuits_benchmark.commands.train.iit import iit_train
from circuits_benchmark.utils.get_cases import get_cases
from circuits_benchmark.commands.train.iit import ioi_train

def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("train")
  run_subparsers = run_parser.add_subparsers(dest="type")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  linear_compression.setup_args_parser(run_subparsers)
  non_linear_compression.setup_args_parser(run_subparsers)
  autoencoder.setup_args_parser(run_subparsers)
  natural_compression.setup_args_parser(run_subparsers)
  iit_train.setup_args_parser(run_subparsers)
  ioi_train.setup_args_parser(run_subparsers)


def run(args):
  training_type = args.type
  if training_type == "ioi":
    ioi_train.run_ioi_training(args)
    return
  for case in get_cases(args):
    print(f"\nRunning training {args.type} on {case}")

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
    elif training_type == "autoencoder":
      train_autoencoder(case, args)
    elif training_type == "natural-compression":
      natural_compression.train_natural_compression(case, args)
    elif training_type == "iit":
      iit_train.run_iit_train(case, args)
    else:
      raise ValueError(f"Unknown training: {training_type}")
