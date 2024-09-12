from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case45(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_double_first_half()

  def get_task_description(self) -> str:
    return "Doubles the first half of the sequence"

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_double_first_half() -> rasp.SOp:
  # Calculate the length of the sequence and store it in a constant sequence.
  length = rasp.SelectorWidth(rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)).named("length")

  # Create a Map operation that turns indices into 1 if they are in the first half, 0 otherwise.
  # Note: We use a trick here by utilizing a SequenceMap that compares indices against half of the length sequence, but as we cannot perform division directly on SOps, we prepare the length beforehand.
  first_half_selector = rasp.SequenceMap(lambda idx, length: 1 if idx < length / 2 else 0, rasp.indices,
                                         length).named("first_half_selector")

  # Apply doubling conditionally: Multiply each element by (1 or 2) based on the first_half_selector.
  # This step combines the original sequence with the selector sequence to apply the doubling only to the first half.
  double_first_half = rasp.SequenceMap(lambda x, sel: x * (1 + sel), rasp.tokens, first_half_selector).named(
    "double_first_half")

  return double_first_half
