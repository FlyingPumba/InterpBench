from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case63(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_count_less_than()

  def get_task_description(self) -> str:
    return "Replaces each element with the number of elements less than it in the sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_count_less_than() -> rasp.SOp:
  # Create a selector that identifies where one element is less than another in the sequence.
  lt_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT).named("lt_selector")

  # Count the number of True comparisons (i.e., number of elements less than) for each element.
  count_lt = rasp.SelectorWidth(lt_selector).named("count_lt")

  return count_lt
