from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case85(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_square_each_element()

  def get_task_description(self) -> str:
    return "Square each element of the input sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_square_each_element() -> rasp.SOp:
  return rasp.Map(lambda x: x ** 2, rasp.tokens).named("square_each_element")
