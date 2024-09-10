import math
from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case50(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_hyperbolic_cosine()

  def get_task_description(self) -> str:
    return "Applies the hyperbolic cosine to each element"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_hyperbolic_cosine() -> rasp.SOp:
  return rasp.Map(math.cosh, rasp.tokens)
