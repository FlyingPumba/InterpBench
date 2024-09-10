import math
from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case55(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_hyperbolic_sine()

  def get_task_description(self) -> str:
    return "Applies the hyperbolic sine to each element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_hyperbolic_sine() -> rasp.SOp:
  hyperbolic_sine = rasp.Map(lambda x: math.sinh(x), rasp.tokens).named("hyperbolic_sine")

  return hyperbolic_sine
