from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case100(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_swap_elements()

  def get_task_description(self) -> str:
    return "Swaps two elements at specified indices (default is 0 and 1)."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False

  def get_max_seq_len(self) -> int:
    return len(self.get_vocab())


def condition(a, idx1, idx2):
  if a == idx2:
    return idx1
  elif a == idx1:
    return idx2
  else:
    return a


def make_swap_elements(index_a=1, index_b=0):
  swaper = rasp.Map(lambda x: condition(x, index_a, index_b), rasp.indices)
  swap_selector = rasp.Select(swaper, rasp.indices, rasp.Comparison.EQ)
  return rasp.Aggregate(swap_selector, rasp.tokens)
