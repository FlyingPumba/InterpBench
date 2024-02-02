#! /usr/bin/env python3
import logging
import sys

import jax

from commands.build_main_parser import build_main_parser
from commands.algorithms import run_algorithm
from commands.compilation import compile_benchmark
from commands.analysis import perform_analysis
from commands.train import train

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
  elif args.command == "analysis":
    perform_analysis.run(args)