import random

import numpy as np
import torch as t

from commands.train import linear_compression, non_linear_compression, autoencoder, natural_compression
from commands.train.autoencoder import train_autoencoder
from commands.train.linear_compression import train_linear_compression
from commands.train.non_linear_compression import train_non_linear_compression
from utils.get_cases import get_cases


def setup_args_parser(subparsers):
  run_parser = subparsers.add_parser("train")
  run_subparsers = run_parser.add_subparsers(dest="type")
  run_subparsers.required = True

  # Setup arguments for each algorithm
  linear_compression.setup_args_parser(run_subparsers)
  non_linear_compression.setup_args_parser(run_subparsers)
  autoencoder.setup_args_parser(run_subparsers)
  natural_compression.setup_args_parser(run_subparsers)


def run(args):
  for case in get_cases(args):
    training_type = args.type
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
    else:
      raise ValueError(f"Unknown training: {training_type}")
