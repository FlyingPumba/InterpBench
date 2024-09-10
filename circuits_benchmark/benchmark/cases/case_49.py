from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case49(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_decrement_to_multiple_of_three()

  def get_task_description(self) -> str:
    return "Decrements each element in the sequence until it becomes a multiple of 3."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_decrement_to_multiple_of_three() -> rasp.SOp:
  return rasp.Map(lambda x: x - x % 3, rasp.tokens).named("decrement_to_multiple_of_three")
