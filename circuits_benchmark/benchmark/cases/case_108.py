from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case108(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_test_at_least_two_equal()

  def get_task_description(self) -> str:
    return "Check if at least two elements in the input list are equal."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_test_at_least_two_equal() -> rasp.SOp:
  equal_selector = rasp.Select(rasp.tokens, rasp.tokens, lambda x, y: x == y)
  checks = rasp.SelectorWidth(equal_selector)
  greater_than_2 = rasp.Select(checks, rasp.Map(lambda x: 2, rasp.indices), rasp.Comparison.GEQ)
  return rasp.Map(lambda x: 1 if x > 0 else 0, rasp.SelectorWidth(greater_than_2))
