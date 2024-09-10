import math
from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case104(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_exponential()

  def get_task_description(self) -> str:
    return "Apply exponential function to all elements of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_exponential() -> rasp.SOp:
  exp_approx = rasp.Map(lambda x: math.exp(x), rasp.tokens)
  return exp_approx