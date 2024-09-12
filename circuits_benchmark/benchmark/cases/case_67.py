from typing import Set

from tracr.rasp import rasp

from circuits_benchmark.benchmark import vocabs
from circuits_benchmark.benchmark.tracr_benchmark_case import TracrBenchmarkCase


class Case67(TracrBenchmarkCase):
  def get_program(self) -> rasp.SOp:
    return make_multiply_by_length()

  def get_task_description(self) -> str:
    return "Multiply each element of the sequence by the length of the sequence."

  def get_vocab(self) -> Set:
    return vocabs.get_int_numbers_vocab()

  def supports_causal_masking(self) -> bool:
    return False


def make_multiply_by_length() -> rasp.SOp:
  # Select all elements by using the TRUE comparison, effectively making every comparison true.
  all_true_selector = rasp.Select(
    rasp.tokens, rasp.tokens, rasp.Comparison.TRUE).named("all_true_selector")

  # The SelectorWidth operation counts the number of true selections, giving us the length of the sequence.
  length_sequence = rasp.SelectorWidth(all_true_selector).named("length_sequence")

  # Use SequenceMap to multiply each element of the original sequence by the length of the sequence.
  multiply_by_length_sequence = rasp.SequenceMap(
    lambda x, y: x * y, rasp.tokens, length_sequence).named("multiply_by_length_sequence")

  return multiply_by_length_sequence
