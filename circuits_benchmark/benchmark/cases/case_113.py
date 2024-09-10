from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case113(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_invert_if_sorted()

  def get_task_description(self) -> str:
    return "Inverts the sequence if it is sorted in ascending order, otherwise leaves it unchanged."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(max=30)

  def supports_causal_masking(self) -> bool:
    return False


def make_invert_if_sorted():
  shifter = rasp.Select(rasp.indices, rasp.indices, lambda x, y: x == y - 1 or (x == 0 and y == 0))
  shifted = rasp.Aggregate(shifter, rasp.tokens)
  checks = rasp.SequenceMap(lambda x, y: 1 if x <= y else 0, shifted, rasp.tokens)
  zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
  invert_decider = rasp.Map(lambda x: 1 if x > 0 else -1, rasp.SelectorWidth(zero_selector))
  avg_idx = rasp.Map(lambda x: x / 2 - 0.5,
                     rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.TRUE)))
  diff_to_avg_idx = rasp.SequenceMap(lambda x, y: x - y, rasp.indices, avg_idx)
  inverter = rasp.SequenceMap(lambda x, y: x + y, avg_idx,
                              rasp.SequenceMap(lambda x, y: x * y, invert_decider, diff_to_avg_idx))
  invert_selector = rasp.Select(inverter, rasp.indices, rasp.Comparison.EQ)
  return rasp.Aggregate(invert_selector, rasp.tokens)
