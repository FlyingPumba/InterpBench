from argparse import Namespace
from typing import List

import pandas as pd
from huggingface_hub import hf_hub_download

from circuits_benchmark.benchmark.benchmark_case import BenchmarkCase
from circuits_benchmark.utils.find_all_subclasses import find_all_transitive_subclasses_in_package


def get_cases(args: Namespace | None = None, indices: List[str] | None = None) -> List[BenchmarkCase]:
  assert (args is None or args.indices is None) or indices is None, "Cannot specify both args.indices and indices"

  classes = find_all_transitive_subclasses_in_package(BenchmarkCase, "circuits_benchmark.benchmark.cases")
  classes = [cls for cls in classes if cls.__name__.startswith("Case")]

  if args is not None and args.indices is not None:
    indices = [idx.lower() for idx in args.indices.split(",")]

  if indices is not None:
    # filter class names that are "CaseN" where N in indices
    classes = [cls for cls in classes if cls.__name__[4:].lower() in indices]

  # sort classes. if id is a number, numerically, otherwise alphabetically
  classes.sort(key=lambda cls:
    cls.__name__[4:] if cls.__name__[4:].isnumeric() else cls.__name__[4:]
  )

  # instantiate all classes found
  return [cls() for cls in classes]

def get_names_of_working_cases() -> list[str]:
  file = hf_hub_download(
    "cybershiptrooper/InterpBench",
    filename="benchmark_cases_metadata.csv",
  )
  df = pd.read_csv(file)
  working_cases = df["case_id"]
  return working_cases.tolist()

def get_names_of_categorical_cases(names_of_cases: list[str]) -> list[str]:
  cases_objs = get_cases(indices=names_of_cases)
  categorical_cases = []
  for case in cases_objs:
      if case.get_hl_model().is_categorical():
          categorical_cases.append(case.get_name())
  return categorical_cases

def get_names_of_regression_cases(cases: list[str]) -> list[str]:
  categorical_cases = get_categorical_cases(cases)
  regression_cases = [case for case in cases if case not in categorical_cases]
  return regression_cases

def get_categorical_cases(cases: list[str]) -> list[BenchmarkCase]:
  cases_objs = get_cases(indices=cases)
  categorical_cases = []
  for case in cases_objs:
      if case.get_hl_model().is_categorical():
          categorical_cases.append(case)
  return categorical_cases

def get_regression_cases(cases: list[str]) -> list[BenchmarkCase]:
  categorical_cases = get_categorical_cases(cases)
  regression_cases = [case for case in cases if case not in categorical_cases]
  return regression_cases