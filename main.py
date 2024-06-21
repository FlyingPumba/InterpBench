#! /usr/bin/env python3
import logging
import sys

import jax

from circuits_benchmark.commands.build_main_parser import build_main_parser
from circuits_benchmark.commands.algorithms import run_algorithm
from circuits_benchmark.commands.compilation import compile_benchmark
from circuits_benchmark.commands.train import train
from circuits_benchmark.commands.evaluation import evaluation

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
  parser = build_main_parser()
  args, _ = parser.parse_known_args(sys.argv[1:])

  if args.command == "compile":
    compile_benchmark.compile_all(args)
  elif args.command == "run":
    run_algorithm.run(args)
  elif args.command == "train":
    train.run(args)
  elif args.command == "eval":
    evaluation.run(args)
