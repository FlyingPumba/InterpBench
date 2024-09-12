import math
from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case73(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sine()

  def get_task_description(self) -> str:
    return "Apply the sine function to each element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_sine() -> rasp.SOp:
  return rasp.Map(lambda x: math.sin(x), rasp.tokens).named("sine")
