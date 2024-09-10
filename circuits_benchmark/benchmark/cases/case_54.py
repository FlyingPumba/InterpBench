from typing import Set, Sequence

import math
from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case54(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_hyperbolic_tangent()

  def get_task_description(self) -> str:
    return "Applies the hyperbolic tangent to each element."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_hyperbolic_tangent() -> rasp.SOp:
  return rasp.Map(math.tanh, rasp.tokens)
