import math
from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case121(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_arcsine()

  def get_task_description(self) -> str:
    return "Compute arcsine of all elements in the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_float_numbers_vocab(min=-1, max=1)


def make_arcsine():
  return rasp.Map(lambda x: math.asin(x), rasp.tokens)
