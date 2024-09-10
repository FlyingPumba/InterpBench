from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case96(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_remove_duplicates()

  def get_task_description(self) -> str:
    return "Set duplicates to 0, keep the first occurrences unchanged."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab(min=1)

  def supports_causal_masking(self) -> bool:
    return False


def make_remove_duplicates() -> rasp.SOp:
  # Compare each element with every other element for equality
  eq_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ).named("eq_selector")

  # Count the number of elements that each element is equal to (including itself)
  eq_count = rasp.SelectorWidth(eq_selector).named("eq_count")

  # Create an index sequence to ensure that for every element, counts are considered only up to its position (i.e., ignore future duplicates)
  index = rasp.Map(lambda x: x + 1, rasp.indices).named("index")
  adjusted_count = rasp.SequenceMap(lambda count, idx: count if count <= idx else 0, eq_count, index).named(
    "adjusted_count")

  # Identify the first occurrences and duplicates. First occurrences will have an adjusted count of 1, replace others with 0.
  first_or_zero = rasp.Map(lambda x: 1 if x == 1 else 0, adjusted_count).named("first_or_zero")

  # Replace duplicates (indicated by 0) in the original sequence with 0, keep the first occurrences unchanged
  remove_duplicates = rasp.SequenceMap(lambda original, flag: original if flag == 1 else 0, rasp.tokens,
                                       first_or_zero).named("remove_duplicates")

  return remove_duplicates
