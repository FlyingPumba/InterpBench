from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case46(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_decrement()

  def get_task_description(self) -> str:
    return "Decrements each element in the sequence by 1"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_decrement() -> rasp.SOp:
  return rasp.Map(lambda x: x - 1, rasp.tokens).named("decrement")
