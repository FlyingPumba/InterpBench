from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case93(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_swap_odd_index()

  def get_task_description(self) -> str:
    return "Swaps the nth with the n+1th element if n%2==1."
    # Note that this means that the first element will remain unchanged.
    # The second will be swapped with the third and so on

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_swap_odd_index() -> rasp.SOp:
  len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
  swaper = rasp.SequenceMap(
    lambda x, y: x if (x == y - 1 and y % 2 == 0 or x == 0) else (x - 1 if (x + 1) % 2 == 1 else x + 1), rasp.indices,
    len)
  swap_selector = rasp.Select(rasp.indices, swaper, rasp.Comparison.EQ)
  swaped = rasp.Aggregate(swap_selector, rasp.tokens)
  return swaped
