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

  # "compile" command
  compile_parser = subparsers.add_parser("compile")
  compile_parser.add_argument("-i", "--indices", type=str, default=None,
                              help="A list of comma separated indices of the cases to compile. "
                                          "If not specified, all cases will be compiled.")

  # "acdc" command
  run_parser = subparsers.add_parser("run")
  run_parser.add_argument("-i", "--indices", type=str, default=None,
                              help="A list of comma separated indices of the cases to run against. "
                                          "If not specified, all cases will be run.")
  run_parser.add_argument("-a", "--algorithm", type=str, required=True, choices=["acdc"],
                              help="The algorithm to use for running against the specified cases. ")

  args = parser.parse_args()

  if args.command == "compile":
    compile_benchmark.compile(args)
  elif args.command == "run":
    run_algorithm.run(args)