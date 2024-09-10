from typing import Set, Sequence

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase
from tracr.rasp import rasp


class Case58(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_mirror_first_half()

  def get_task_description(self) -> str:
    return "Mirrors the first half of the sequence to the second half."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_mirror_first_half() -> rasp.SOp:
  # Create a selector for all elements (used to calculate the sequence length)
  all_true_selector = rasp.Select(
    rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")
  length = rasp.SelectorWidth(all_true_selector).named("length")

  # Creating a selector that selects the first half of the sequence
  first_half_selector = rasp.Select(
    rasp.indices,
    rasp.Map(lambda x: x // 2, length),
    rasp.Comparison.LT
  ).named("first_half_selector")

  # Creating a selector for reversing the indices for the second half of the sequence
  mirror_selector = rasp.Select(
    rasp.indices,
    rasp.SequenceMap(lambda x, l: l - 1 - x if x >= l // 2 else x, rasp.indices, length),
    rasp.Comparison.EQ
  ).named("mirror_selector")

  # Aggregate using the mirror selector to mirror the first half onto the second half
  mirrored_sequence = rasp.Aggregate(
    mirror_selector,
    rasp.tokens,
    default=None
  ).named("mirrored_sequence")

  return mirrored_sequence
