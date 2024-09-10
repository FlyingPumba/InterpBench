import math
from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case70(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_cosine()

  def get_task_description(self) -> str:
    return "Apply the cosine function to each element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_cosine() -> rasp.SOp:
  return rasp.Map(lambda x: math.cos(x), rasp.tokens).named("apply_cosine")