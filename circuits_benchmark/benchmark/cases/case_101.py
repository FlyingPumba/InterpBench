from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case101(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_square()

  def get_task_description(self) -> str:
    return "Check if each element is a square of an integer."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(max=30)


def make_check_square() -> rasp.SOp:
  return rasp.Map(lambda x: 1 if x ** 0.5 == int(x ** 0.5) else 0, rasp.tokens)
