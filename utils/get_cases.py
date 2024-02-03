from argparse import Namespace
from typing import List

from benchmark.benchmark_case import BenchmarkCase
from utils.find_all_subclasses import find_all_subclasses_in_package

BENCHMARK_DIR = "benchmark"


def get_cases(args: Namespace | None = None) -> List[BenchmarkCase]:
  classes = find_all_subclasses_in_package(BenchmarkCase, "benchmark.cases")

  if args is not None and args.indices is not None:
    # filter class names that are "CaseN" where N in indices
    classes = [cls for cls in classes if cls.__name__[4:] in args.indices.split(",")]

  # instantiate all classes found
  return [cls() for cls in classes]
