from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case78(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_count_occurrences()

  def get_task_description(self) -> str:
    return "Count the occurrences of each element in a sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_ascii_letters_vocab(count=10)

  def supports_causal_masking(self) -> bool:
    return False


def make_count_occurrences() -> rasp.SOp:
  """
  Creates an SOp that transforms a sequence so each element is replaced by the number of times it appears in the sequence.
  """
  # Selector that compares each element with every other element to find duplicates
  eq_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("eq_selector")

  # Counts the occurrences of each element based on the equality comparison
  count_occurrences = rasp.SelectorWidth(eq_selector).named("count_occurrences")

  return count_occurrences
