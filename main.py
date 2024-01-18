#! /usr/bin/env python3
import logging

import jax

from build_main_parser import build_main_parser
from commands import compile_benchmark, run_algorithm

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
logging.basicConfig(level=logging.ERROR)

if __name__ == "__main__":
  parser = build_main_parser()
  args = parser.parse_args()

  if args.command == "compile":
    compile_benchmark.compile_all(args)
  elif args.command == "run":
    run_algorithm.run(args)