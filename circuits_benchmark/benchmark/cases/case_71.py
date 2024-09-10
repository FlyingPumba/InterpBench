from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case71(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_divide_by_length()

  def get_task_description(self) -> str:
    return "Divide each element by the length of the sequence"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_divide_by_length() -> rasp.SOp:
  # Step 1: Create a selector that selects all elements (TRUE condition)
  all_true_selector = rasp.Select(
    rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")

  # Calculate the length of the sequence using SelectorWidth
  length = rasp.SelectorWidth(all_true_selector).named("length")

  # Step 2: Divide each element by the length of the sequence
  divided_by_length = rasp.SequenceMap(lambda x, length: x / length if length > 0 else 0, rasp.tokens, length).named(
    "divided_by_length")

  return divided_by_length
