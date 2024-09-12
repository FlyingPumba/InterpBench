from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case65(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_cube_root()

  def get_task_description(self) -> str:
    return "Calculate the cube root of each element in the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_cube_root() -> rasp.SOp:
  # Define the cube root operation to be applied to each element.
  cube_root_operation = rasp.Map(lambda x: x ** (1 / 3) if x >= 0 else -(-x) ** (1 / 3), rasp.tokens).named(
    "cube_root_operation")

  return cube_root_operation
