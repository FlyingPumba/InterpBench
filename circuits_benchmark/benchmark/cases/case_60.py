from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case60(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_increment()

  def get_task_description(self) -> str:
    return "Increment each element in the sequence by 1."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()


def make_increment() -> rasp.SOp:
  return rasp.Map(lambda x: x + 1, rasp.tokens).named("increment")
