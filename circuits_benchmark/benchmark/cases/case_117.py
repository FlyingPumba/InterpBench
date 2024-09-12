from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case117(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_sum_of_last_two()

  def get_task_description(self) -> str:
    return "Given a list of integers, return the sum of the last two elements."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_sum_of_last_two() -> rasp.SOp:
  len = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE))
  last_idx = rasp.Map(lambda x: x - 1, len)
  second_to_last_idx = rasp.Map(lambda x: x - 2, len)
  last_elt = rasp.Aggregate(rasp.Select(rasp.indices, last_idx, rasp.Comparison.EQ), rasp.tokens)
  second_to_last_elt = rasp.Aggregate(rasp.Select(rasp.indices, second_to_last_idx, rasp.Comparison.EQ), rasp.tokens)
  return rasp.SequenceMap(lambda x, y: x + y, last_elt, second_to_last_elt)
