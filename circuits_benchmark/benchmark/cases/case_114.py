import math
from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case114(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_logarithm()

  def get_task_description(self) -> str:
    return "Apply a logarithm base 10 to each element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=1)


def make_logarithm() -> rasp.SOp:
  def apply_log(element):
    return math.log(element, 10)

  # Applying the placeholder logarithm function to each element
  return rasp.Map(apply_log, rasp.tokens).named("logarithm")
