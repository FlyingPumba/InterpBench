import argparse

from commands import compile_benchmark, run_algorithm


def build_main_parser():
  # define commands for our main script.
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest="command")
  subparsers.required = True

  # Setup command arguments
  compile_benchmark.setup_args_parser(subparsers)
  run_algorithm.setup_args_parser(subparsers)

  return parser