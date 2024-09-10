from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case90(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_identity()

  def get_task_description(self) -> str:
    return "Identity"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def is_trivial(self) -> bool:
    return True


def make_identity() -> rasp.SOp:
  return rasp.Map(lambda x: x, rasp.tokens)
