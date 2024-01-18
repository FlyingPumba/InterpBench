#! /usr/bin/env python3
import argparse
import jax
import logging

from commands import compile_benchmark, run_algorithm

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
  # define commands for our main script.
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command")
  subparsers.required = True

  # Setup command arguments
  compile_benchmark.setup_args_parser(subparsers)
  run_algorithm.setup_args_parser(subparsers)

  args = parser.parse_args()

  if args.command == "compile":
    compile_benchmark.compile(args)
  elif args.command == "run":
    run_algorithm.run(args)