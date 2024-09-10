from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case52(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_square_root()

  def get_task_description(self) -> str:
    return "Takes the square root of each element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_square_root() -> rasp.SOp:
  return rasp.Map(lambda x: x ** 0.5, rasp.tokens).named("square_root")
