from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case129(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_check_multiple_of_n()

  def get_task_description(self) -> str:
    return "Checks if all elements are a multiple of n (set the default at 2)."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_check_multiple_of_n(n=2) -> rasp.SOp:
  checks = rasp.Map(lambda x: 1 if x % n == 0 else 0, rasp.tokens).named(f"check_multiple_of_{n}")
  zero_selector = rasp.Select(checks, rasp.Map(lambda x: 0, rasp.indices), rasp.Comparison.EQ)
  return rasp.Map(lambda x: 0 if x > 0 else 1, rasp.SelectorWidth(zero_selector))
