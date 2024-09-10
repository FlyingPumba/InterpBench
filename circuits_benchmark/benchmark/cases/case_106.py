from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case106(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_mask_sequence()

  def get_task_description(self) -> str:
    return "Sets all elements to zero except for the element at index 1."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def is_trivial(self) -> bool:
    return True


def make_mask_sequence(index=1) -> rasp.SOp:
  return rasp.SequenceMap(lambda x, y: x if y == index else 0, rasp.tokens, rasp.indices)
