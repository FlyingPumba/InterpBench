from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case74(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_interleave_reverse()

  def get_task_description(self) -> str:
    return "Interleaves elements with their reverse order Numbers at the odd indices should be in reverse order."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_interleave_reverse():
  len = rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE))
  shifter = rasp.SequenceMap(lambda x, y: x if x % 2 == 0 else (y - x if y % 2 == 0 else y - x - 1), rasp.indices,
                             len)
  shift_selector = rasp.Select(rasp.indices, shifter, rasp.Comparison.EQ)
  return rasp.Aggregate(shift_selector, rasp.tokens)

